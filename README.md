# SPG-CDENet
Official Code: SPG-CDENet – Spatial Prior-Guided Cross Dual Encoder Network for Multi-Organ Segmentation.
<img width="3059" height="1349" alt="overview" src="https://github.com/user-attachments/assets/c988ced3-2ba2-4937-820d-0a2aef6e8c1b" />

## Training and Evaluation On Synapse Dataset

CUDA_VISIBLE_DEVICES=0  python train.py --fusion_type crossattn --dataset Synapse

CUDA_VISIBLE_DEVICES=0  python test.py --fusion_type crossattn --dataset Synapse

The pre-trained models can be available at https://drive.google.com/file/d/1IsWJ6JlQ7-x78y6XpTrkCtS5QC_xVPdL/view?usp=drive_link.

## Training and Evaluation On ACDC Dataset
CUDA_VISIBLE_DEVICES=0  python train.py --fusion_type crossattn --dataset ACDC

CUDA_VISIBLE_DEVICES=0  python test.py --fusion_type crossattn --dataset ACDC

The pre-trained models can be available at https://drive.google.com/file/d/1JRLRcPjIdgkBSHGzOZyRqBAlMmFBowHL/view?usp=sharing.


##  Quantitative results

Quantitative results On Synapse
<img width="1299" height="549" alt="image" src="https://github.com/user-attachments/assets/70b87922-b5a1-45f6-af9c-98f3158ebd71" />

Quantitative results On ACDC
<img width="631" height="420" alt="image" src="https://github.com/user-attachments/assets/4215ecb0-d65f-41a8-893f-516f592b924f" />


