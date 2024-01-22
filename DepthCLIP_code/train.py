# -*- coding:utf-8 -*-
"""
# @Author  :Shan Huang
# @Time    :2023/12/26 20:07
# @File    :train.py
"""
import os
import torch
import logging
import argparse
import torch.utils.data
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from torchvision.transforms.functional import to_pil_image

from calculate_error import *
from logger import AverageMeter
from datasets.datasets_list import MyDataset
from dpt.models import DPTDepthModel
from simpleDepthClip import *
from utils import *
from model import *
import torch.optim.lr_scheduler as lr_scheduler

depth_templates = ['This {} is {}']
obj_classes = ['object']
depth_classes = ['giant', 'extremely close', 'close', 'not in distance', 'a little remote', 'far', 'unseen']
bin_list = [1.00, 1.50, 2.00, 2.25, 2.50, 2.75, 3.00]
temperature = 0.1
clip_vis = 'RN50'

parser = argparse.ArgumentParser(
    description='Transformer-based Monocular Depth Estimation with Attention Supervision',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Directory setting
parser.add_argument('--models_list_dir', type=str, default='')
parser.add_argument('--result_dir', type=str, default='')
parser.add_argument('--model_dir', type=str)
parser.add_argument('--other_method', type=str, default='MonoCLIP')
parser.add_argument('--trainfile_kitti', type=str, default="./datasets/eigen_train_files_with_gt_dense.txt")
parser.add_argument('--testfile_kitti', type=str, default="./datasets/eigen_test_files_with_gt_dense.txt")
parser.add_argument('--trainfile_nyu', type=str,
                    default=r"/home/student/DepthCLIP/DepthCLIP_code/datasets/nyudepthv2_train_files_with_gt_dense.txt")
parser.add_argument('--testfile_nyu', type=str,
                    default=r"/home/student/DepthCLIP/DepthCLIP_code/datasets/nyudepthv2_test_files_with_gt_dense.txt")
parser.add_argument('--data_path', type=str,
                    default=r"/home/student/DepthCLIP/DepthCLIP_code/datasets/NYU_Depth_V2/official_splits")
parser.add_argument('--use_dense_depth', action='store_true', help='using dense depth data for gradient loss')

# Dataloader setting
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N', help='number of data loading workers')
parser.add_argument('--epoch_size', default=0, type=int, metavar='N',
                    help='manual epoch size (will match dataset size if not set)')

parser.add_argument('--lr', default=0, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--batch_size', default=16, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--seed', default=0, type=int, help='seed for random functions, and network initialization')
parser.add_argument('--dataset', type=str, default="NYU")

# Logging setting
parser.add_argument('--print-freq', default=100, type=int, metavar='N', help='print frequency')
parser.add_argument('--log-metric', default='_LRDN_evaluation.csv', metavar='PATH',
                    help='csv where to save validation metric value')
parser.add_argument('--val_in_train', action='store_true', help='validation process in training')

# Model setting
parser.add_argument('--encoder', type=str, default="ResNext101")
parser.add_argument('--norm', type=str, default="BN")
parser.add_argument('--act', type=str, default="ReLU")
parser.add_argument('--height', type=int, default=352)
parser.add_argument('--width', type=int, default=704)
parser.add_argument('--max_depth', default=80.0, type=float, metavar='MaxVal', help='max value of depth')
parser.add_argument('--lv6', action='store_true', help='use lv6 Laplacian decoder')

# Train setting
parser.add_argument('--epochs', default=100, type=int)

# Evaluation setting
parser.add_argument('--evaluate', action='store_true', help='evaluate score')
parser.add_argument('--multi_test', action='store_true', help='test all of model in the dir')
parser.add_argument('--img_save', action='store_true', help='will save test set image')
parser.add_argument('--cap', default=80.0, type=float, metavar='MaxVal', help='cap setting for kitti eval')

# GPU parallel process setting
parser.add_argument('--gpu_num', type=str, default="0,1,2,3", help='force available gpu index')
parser.add_argument('--rank', type=int, help='node rank for distributed training', default=0)


logging.basicConfig(filename='training_log4', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_attention_loss(volume, gt):
    cos = torch.nn.CosineSimilarity(dim=3, eps=1e-08)
    batch = volume.shape[0]
    dim = volume.shape[2]
    losses3 = torch.sum((gt - volume).abs()) / (batch * dim * dim * 1)
    part1 = torch.sum(torch.abs(1 - cos(volume, gt)))
    part2 = torch.sum(torch.abs(1 - cos(volume.transpose(2, 3), gt.transpose(2, 3))))
    losses3 = losses3 + (part1 + part2) / (batch * dim * 1)
    return losses3


def get_depth_volume(depth, level, normalize=False):
    batch, H, W = depth.shape
    if normalize:
        temp = torch.flatten(depth, 1)
        max_depth = torch.max(temp, 1)[0].unsqueeze(1)
        temp = temp / max_depth
    else:
        temp = torch.flatten(depth, 1)
    attention = torch.zeros((batch, 1, H * W, H * W)).cuda()
    volume = torch.abs(temp.unsqueeze(2) - temp.unsqueeze(1))
    attention[:, 0, :, :] = torch.softmax((level) * (-volume), 2)
    return attention


def main():
    args = parser.parse_args()
    print('=> Index of using GPU: ', args.gpu_num)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num
    assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."

    torch.manual_seed(args.seed)
    if args.evaluate is True:
        save_path = save_path_formatter(args, parser)
        args.save_path = 'checkpoints' / save_path

    ######################   Data loading part    ##########################
    if args.dataset == 'KITTI':
        args.max_depth = 80.0
    elif args.dataset == 'NYU':
        args.max_depth = 10.0

    if args.result_dir == '':
        args.result_dir = './' + args.dataset + '_Eval_results'
    args.log_metric = args.dataset + '_' + args.encoder + args.log_metric

    train_set = MyDataset(args, train=True)
    val_set = MyDataset(args, train=False)
    print("=> Dataset: ", args.dataset)
    print("=> Data height: {}, width: {} ".format(args.height, args.width))
    print('=> test  samples_num: {}  '.format(len(train_set)))

    test_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=True, sampler=test_sampler)
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=1, shuffle=False,
        num_workers=0, pin_memory=True, sampler=test_sampler
    )

    cudnn.benchmark = True
    ###########################################################################

    ###################### setting model list #################################
    if args.multi_test is True:
        print("=> all of model tested")
        models_list_dir = Path(args.models_list_dir)
        models_list = sorted(models_list_dir.files('*.pkl'))
    else:
        print("=> just one model tested")
        models_list = [args.model_dir]
    ###########################################################################

    ###################### setting Network part ###################
    print("=> creating model")
    if args.other_method == None:
        Model = LDRN(args)

        num_params_encoder = 0
        num_params_decoder = 0
        for p in Model.encoder.encoder.parameters():
            num_params_encoder += p.numel()
        for p in Model.decoder.parameters():
            num_params_decoder += p.numel()
        print("===============================================")
        print("model encoder parameters: ", num_params_encoder)
        print("model decoder parameters: ", num_params_decoder)
        print("Total parameters: {}".format(num_params_encoder + num_params_decoder))
        print("===============================================")
    else:
        if args.other_method == 'DPT-Large':
            Model = DPTDepthModel(
                scale=0.000305,
                shift=0.1378,
                invert=False,
                backbone="vitl16_384",
                non_negative=True,
                enable_attention_hooks=False, )
        if args.other_method == 'Adabins':
            from Adabins import UnetAdaptiveBins
            if args.dataset == 'KITTI':
                Model = UnetAdaptiveBins.build(n_bins=256, min_val=1e-3, max_val=80, norm="linear")
            if args.dataset == 'NYU':
                Model = UnetAdaptiveBins.build(n_bins=256, min_val=1e-3, max_val=10, norm="linear")

        if args.other_method == 'MonoCLIP':
            if args.dataset == 'KITTI':
                Model = IntergratedDepthCLIP()
                Model.unfixedvalue.conv2d.weight.data = torch.tensor([1, 1, 1, 1, 1, 1, 1]).view(
                    [1, 7, 1, 1]).cuda().half()
            if args.dataset == 'NYU':
                Model = IntergratedDepthCLIP()
                Model.unfixedvalue.conv2d.weight.data = torch.tensor([1, 1, 1, 1, 1, 1, 1]).view([1, 7, 1, 1]).cuda().half()

        num_params = 0
        for p in Model.parameters():
            num_params += p.numel()
        print("===============================================")
        print("Total parameters: {}".format(num_params))
        print("===============================================")
    Model = Model.cuda()
    # Model = torch.nn.DataParallel(Model)

    ###################### setting validate part ###################
    if args.dataset == 'KITTI':
        error_names = ['abs_diff', 'abs_rel', 'sq_rel', 'a1', 'a2', 'a3', 'rmse', 'rmse_log']
    elif args.dataset == 'NYU':
        # error_names = ['abs_diff', 'abs_rel', 'log10', 'a1', 'a2', 'a3','rmse','rmse_log']
        error_names = ['abs_diff', 'a1', 'a2', 'a3', 'abs_rel', 'log10', 'rmse']
    elif args.dataset == 'Make3D':
        error_names = ['abs_diff', 'abs_rel', 'ave_log10', 'rmse']
    errors = AverageMeter(i=len(error_names))
    length = len(val_loader)
    ###########################################################################

    # ###################### setting training part ###################
    # optimizer = torch.optim.Adam(Model.unfixedvalue.parameters(), lr=0.1)
    # scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.75)
    # scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=1)
    # criterion = DepthLoss()
    # optimizer = torch.optim.Adam(Model.unfixedvalue.parameters(), lr=0.1)
    optimizer = torch.optim.SGD(Model.unfixedvalue.parameters(), lr=0.1)
    # scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.75)
    scheduler = lr_scheduler.StepLR(optimizer,step_size=10, gamma=0.5)
    criterion = torch.nn.MSELoss()




    best_loss = float('inf')
    best_model = None
    ###########################################################################

    for epoch in range(args.epochs):
        loop = tqdm(enumerate(train_loader),total=len(train_loader))
        for index, (rgb_data, gt_data, dense) in loop:
            Model.train()
            if gt_data.ndim != 4 and gt_data[0] == False:
                continue
            rgb_data = rgb_data.cuda().half()  # [4 3 416 544]
            gt_data = gt_data.cuda().half() # [4 1 416 544]
            input_img = rgb_data

            optimizer.zero_grad()
            output_depth = Model(input_img)
            loss = criterion(output_depth, gt_data)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                for param in Model.unfixedvalue.parameters():
                    if param.grad is not None:
                        param.grad.data = torch.clamp(param.grad.data, min=0.0)
                Model.unfixedvalue.binlist.data = torch.sort(Model.unfixedvalue.binlist.data).values
            lossvalue, lrvalue=loss.item(), scheduler.get_last_lr()
            loop.set_description(f'Train Epoch [{epoch}/{args.epochs}]')
            loop.set_postfix({"Loss":lossvalue, "Learning Rate":lrvalue})
        scheduler.step()
        Model.eval()

        with torch.no_grad():
            loop_test = tqdm(enumerate(val_loader),total=len(val_loader))
            for i, (rgb_data, gt_data, dense) in loop_test:
                rgb_data = rgb_data.cuda()
                gt_data = gt_data.cuda()
                input_img = rgb_data
                output_depth = Model(input_img)

                if args.dataset == 'KITTI':
                    err_result = compute_errors(gt_data, output_depth, crop=True, cap=args.cap)
                elif args.dataset == 'NYU':
                    err_result = compute_errors_NYU(gt_data, output_depth, crop=True, idx=i)

                errors.update(err_result)
                errors_value = errors.avg
                error_string = ', '.join('{} : {:.3f}'.format(name, error) for name, error in
                                         zip(error_names[0:len(error_names)], errors_value[0:len(errors_value)]))
                loop_test.set_description(f'Validation Epoch [{epoch}/{args.epochs}]')
                loop_test.set_postfix({"error":error_string})

        errors_value = errors.avg
        error_string = ', '.join('{} : {:.3f}'.format(name, error) for name, error in zip(error_names[0:len(error_names)], errors_value[0:len(errors_value)]))

        print(Model.unfixedvalue.binlist.data)
        print('\n')
        print(Model.unfixedvalue.conv2d.weight.data)
        # print(f'Epoch [{epoch + 1}/{args.epochs}], Loss: {loss.item()}, LR: {scheduler.get_last_lr()}')
        # print(' * Avg {}\n'.format(error_string))
        logging.info(
            f'Epoch [{epoch + 1}/{args.epochs}], Learning Rate: {optimizer.param_groups[0]["lr"]}, Loss: {loss.item()},Errors:{error_string}')
    unfixedvalue_param = Model.unfixedvalue.state_dict()
    torch.save(unfixedvalue_param, 'modelparam/TRAIN_MODEL_4.pth')


if __name__ == '__main__':
    main()
