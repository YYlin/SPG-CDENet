import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets.dataset_synapse import Synapse_dataset
from datasets.dataset_acdc import BaseDataSets as ACDC_dataset
from utils import test_single_volume
from networks.DE_UNET import DENet_Cross_Dual_Encoder


parser = argparse.ArgumentParser()
parser.add_argument('--volume_path', type=str,
                    default='./dataset/Synapse/', help='root dir for validation volume data')  # for acdc volume_path=root_dir
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')
parser.add_argument('--num_classes', type=int,
                    default=9, help='output channel of network')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')
parser.add_argument('--output_dir', type=str, 
                    default='./CKPT',help='output dir')
parser.add_argument('--type_unet', type=str,
                    default='DENet_Cross_Dual_Encoder', choices=['DENet_Cross_Dual_Encoder'], help='output dir')
parser.add_argument('--max_iterations', type=int,default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int, default=400, help='maximum epoch number to train')
parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')
parser.add_argument('--is_savenii', action="store_true", help='whether to save results during inference')
parser.add_argument('--test_save_dir', type=str, default='./MISSFormer_Result', help='saving prediction as nii!')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.05, help='segmentation network learning rate')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
# parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                            'full: cache all data, '
                            'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', default=True, help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true', help='Test throughput only')
parser.add_argument('--output_both_stage', type=bool,
                    default=True, help='batch_size per gpu')

parser.add_argument('--vit_name', type=str,
                    default='R50-ViT-B_16', choices=['R50-ViT-B_16', 'ViT-B_16'], help='select one vit model')
parser.add_argument('--n_skip', type=int, default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_patches_size', type=int,
                    default=16, help='vit_patches_size, default is 16')

parser.add_argument('--fusion_type', type=str,
                    default='crossattn', choices=['none', 'sum', 'concat', 'crossattn', 'single_crossattn'])
args = parser.parse_args()

if args.dataset == "Synapse":
    args.volume_path = os.path.join(args.volume_path, "test_vol_h5")
if args.dataset == "ACDC":
    args.volume_path = './dataset/ACDC'


def inference(tmp_snapshot, args, model, test_save_path=None):
    if args.dataset == 'Synapse':
        db_test = args.Dataset(base_dir=args.volume_path, split="test_vol",img_size=args.img_size, list_dir=args.list_dir)
    else:
        db_test = args.Dataset(base_dir=args.volume_path, split="test_vol")
   
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    metric_list = 0.0
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        h, w = sampled_batch["image"].size()[2:]
        image, label, case_name, organ = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0], sampled_batch[
                'organ']
        
        metric_i = test_single_volume(organ, image, label, model, classes=args.num_classes, patch_size=[args.img_size, args.img_size],
                                    test_save_path=test_save_path, case=case_name, z_spacing=args.z_spacing)
        metric_list += np.array(metric_i)
        
        logging.info('idx %d case %s mean_dice %f mean_hd95 %f' % (i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
    metric_list = metric_list / len(db_test)
    for i in range(1, args.num_classes):
        logging.info('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i-1][0], metric_list[i-1][1]))
    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    logging.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))
    # logging.info('tmp_snapshot:', tmp_snapshot, 'Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))

    result_file = os.path.join("test_result_summary1.txt")
    with open(result_file, "a") as f:
        f.write("Snapshot: {}\n".format(tmp_snapshot))
        f.write("Mean Dice: {:.4f}\n".format(performance))
        f.write("Mean HD95: {:.4f}\n".format(mean_hd95))
        f.write("-" * 40 + "\n")

    return "Testing Finished!"

if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_config = {
        'Synapse': {
            'Dataset': Synapse_dataset,
            'volume_path': args.volume_path,
            'list_dir': './lists/lists_Synapse',
            'num_classes': 9,
            'z_spacing': 1,
        },
        'ACDC': {
            'Dataset': ACDC_dataset,  # datasets.dataset_acdc.BaseDataSets,
            'volume_path': './dataset/ACDC',
            'list_dir': None,
            'num_classes': 4,
            'z_spacing': 5,
            'info': '3D'
        },
    }
    dataset_name = args.dataset
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.volume_path = dataset_config[dataset_name]['volume_path']
    args.Dataset = dataset_config[dataset_name]['Dataset']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.z_spacing = dataset_config[dataset_name]['z_spacing']
    args.is_pretrain = True

    if args.type_unet == 'MISSFormer':
        net = MISSFormer(num_classes=args.num_classes).cuda(0)
    elif args.type_unet == 'UNET':
        net = Unet(args.num_classes).cuda(0)
    elif args.type_unet == 'Resnet50UNet':
        net = Resnet50UNet(num_classes=args.num_classes).cuda(0)
    elif args.type_unet == 'AttU_Net':
        net = AttU_Net(args.num_classes).cuda(0)
    elif args.type_unet == 'Resnet50_Atten_UNet':
        net = Resnet50UNet_Attention(args.num_classes).cuda(0)
    elif args.type_unet == 'R2U_Net':
        net = R2U_Net(args.num_classes).cuda(0)
    elif args.type_unet == 'FCT':
        net = FCT(args.num_classes).cuda(0)
    elif args.type_unet == 'DENet_Single':
        net = DENet_Single(args.num_classes).cuda(0)
    elif args.type_unet == 'DENet_Organ':
        net = DENet_Organ(args.num_classes).cuda(0)
    elif args.type_unet == 'DENet_Cross_Dual_Encoder':
        net = DENet_Cross_Dual_Encoder(num_classes=args.num_classes, fusion_type=args.fusion_type).cuda(0)
    elif args.type_unet == 'TransUNet':
        from networks_transunet.vit_seg_modeling import VisionTransformer as ViT_seg
        from networks_transunet.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg

        config_vit = CONFIGS_ViT_seg[args.vit_name]
        config_vit.n_classes = args.num_classes
        config_vit.n_skip = args.n_skip
        if args.vit_name.find('R50') != -1:
            config_vit.patches.grid = (
                int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
        net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()
        net.load_from(weights=np.load(config_vit.pretrained_path))
    elif args.type_unet == 'SwinUNet':
        from networks_swinunet.vision_transformer import SwinUnet as ViT_seg
        from networks_swinunet.config import get_config

        config = get_config(args)
        net = ViT_seg(config, img_size=args.img_size, num_classes=args.num_classes).cuda()

    for tmp_snapshot in os.listdir(args.output_dir):
            
            if tmp_snapshot.endswith(".pth"):
                 
                 snapshot = os.path.join(args.output_dir, tmp_snapshot)
                 msg = net.load_state_dict(torch.load(snapshot))

                 print("self trained swin unet", msg)
                 snapshot_name = snapshot.split('/')[-1].split('.')[0]

                 log_folder = './test_log/test_log_' + snapshot_name
                 os.makedirs(log_folder, exist_ok=True)
            # 清除之前的所有 handler
                 for handler in logging.root.handlers[:]:
                     logging.root.removeHandler(handler)

            # 重新设置日志配置
                 logging.basicConfig(
                     filename=os.path.join(log_folder, snapshot_name + ".txt"),
                     level=logging.INFO,
                     format='[%(asctime)s.%(msecs)03d] %(message)s',
                     datefmt='%H:%M:%S'
                 )
            # 添加新的 stdout handler
                 logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
                 logging.info(snapshot_name)

                 if args.is_savenii:
                     args.test_save_dir = os.path.join(args.output_dir, "predictions_" + snapshot_name)
                     test_save_path = args.test_save_dir
                     os.makedirs(test_save_path, exist_ok=True)
                 else:
                     test_save_path = None
                 inference(tmp_snapshot, args, net, test_save_path)


   

