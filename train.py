import argparse
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from networks.DE_UNET import DENet_Cross_Dual_Encoder
from trainer import trainer_synapse, trainer_acdc

import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse', help='experiment_name')
parser.add_argument('--num_classes', type=int,
                    default=9, help='output channel of network')
parser.add_argument('--output_dir', type=str, 
                    default='./CKPT',help='output dir')
parser.add_argument('--type_unet', type=str, 
                    default='DENet_Cross_Dual_Encoder', choices=['DENet_Cross_Dual_Encoder'], help='output dir')
parser.add_argument('--max_iterations', type=int,
                    default=90000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=400, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=24, help='batch_size per gpu')
parser.add_argument('--output_both_stage', type=bool,
                    default=False, help='batch_size per gpu')

parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.0005,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
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
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true', help='Test throughput only')
parser.add_argument('--cfg', type=str, required=False, default='networks_swinunet/swin_tiny_patch4_window7_224_lite.yaml', metavar="FILE", help='path to config file', )
parser.add_argument('--volume_path', type=str,
                    default='./dataset/Synapse/', help='root dir for validation volume data')

parser.add_argument('--vit_name', type=str,
                    default='R50-ViT-B_16', choices=['R50-ViT-B_16', 'ViT-B_16'], help='select one vit model')
parser.add_argument('--n_skip', type=int, default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_patches_size', type=int,
                    default=16, help='vit_patches_size, default is 16')

parser.add_argument('--resume_path', type=str, help='resume from checkpoint')
parser.add_argument('--fusion_type', type=str, default='crossattn', choices=['none', 'sum', 'concat', 'crossattn', 'single_crossattn'])
args = parser.parse_args()
if args.dataset == "Synapse":
    args.root_path = './dataset/Synapse/train_npz'
    args.num_classes = 9
    args.volume_path = os.path.join(args.volume_path, "test_vol_h5")
elif args.dataset == "ACDC":
    args.root_path = './dataset/ACDC' 
    args.num_classes = 4
    args.volume_path = './dataset/ACDC'

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

    if args.batch_size != 24 and args.batch_size % 5 == 0:
        args.base_lr *= args.batch_size / 24

    args.output_dir =os.path.join(args.output_dir, args.dataset + str(args.batch_size) + '_' + args.type_unet + '_' + args.fusion_type)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    net = DENet_Cross_Dual_Encoder(num_classes=args.num_classes, fusion_type=args.fusion_type).cuda(0)

    if args.resume_path:
        msg = net.load_state_dict(torch.load(args.resume_path, map_location='cuda:0'))

    trainer = {'Synapse': trainer_synapse,'ACDC': trainer_acdc,}

    trainer[args.dataset](args, net, args.output_dir)