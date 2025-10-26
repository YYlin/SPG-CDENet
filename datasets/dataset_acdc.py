import itertools
import os
import random
import re
from glob import glob

import cv2
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from skimage import io
import imgaug as ia
import imgaug.augmenters as iaa  # 导入iaa


class BaseDataSets(Dataset):
    def __init__(self, base_dir=None, img_size=224, split='train', transform=None, y_transform=None):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.img_size=img_size
        self.x_transform = transform
        self.y_transform = y_transform

        train_ids, val_ids, test_ids = self._get_ids()
        if self.split.find('train') != -1:
            self.all_slices = os.listdir(self._base_dir + "/ACDC_training_slices")
            for ids in train_ids:
                new_data_list = list(filter(lambda x: re.match('{}.*'.format(ids), x) != None, self.all_slices))
                self.sample_list.extend(new_data_list)

        elif self.split.find('val') != -1:
            self.all_volumes = os.listdir(
                self._base_dir + "/ACDC_training_volumes")
            for ids in val_ids:
                new_data_list = list(filter(lambda x: re.match('{}.*'.format(ids), x) != None, self.all_volumes))
                self.sample_list.extend(new_data_list)

        elif self.split.find('test') != -1:
            self.all_volumes = os.listdir(
                self._base_dir + "/ACDC_training_volumes")
            for ids in test_ids:
                new_data_list = list(filter(lambda x: re.match('{}.*'.format(ids), x) != None, self.all_volumes))
                self.sample_list.extend(new_data_list)

        self.img_aug = iaa.SomeOf((0,4),[
            iaa.Flipud(0.5, name="Flipud"),
            iaa.Fliplr(0.5, name="Fliplr"),
            iaa.AdditiveGaussianNoise(scale=0.005 * 255),
            iaa.GaussianBlur(sigma=(1.0)),
            iaa.LinearContrast((0.5, 1.5), per_channel=0.5),
            iaa.Affine(scale={"x": (0.5, 2), "y": (0.5, 2)}),
            iaa.Affine(rotate=(-40, 40)),
            iaa.Affine(shear=(-16, 16)),
            iaa.PiecewiseAffine(scale=(0.008, 0.03)),
            iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)})
        ], random_order=True)

        print("total {} samples".format(len(self.sample_list)))

    def _get_ids(self):
        all_cases_set = ["patient{:0>3}".format(i) for i in range(1, 101)]
        testing_set = ["patient{:0>3}".format(i) for i in range(1, 21)]
        validation_set = ["patient{:0>3}".format(i) for i in range(21, 31)]
        training_set = [i for i in all_cases_set if i not in testing_set+validation_set]
        return [training_set, validation_set, testing_set]

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]

        if self.split == "train":
            h5f = h5py.File(self._base_dir + "/ACDC_training_slices/{}".format(case), 'r')
            image = h5f['image'][:]
            label = h5f['label'][:]  # fix sup_type to label

            h5f_o = h5py.File("./dataset/ACDC_SwinUNet_Organ/ACDC_training_slices/{}".format(case), 'r')
            organ = h5f_o['organ'][:]

            image, label, organ = augment_seg(self.img_aug, image, label, organ)
            x, y = image.shape
            x_o, y_o = organ.shape
            if x != self.img_size or y != self.img_size:
                image = zoom(image, (self.img_size / x, self.img_size / y), order=3)  # why not 3?
                label = zoom(label, (self.img_size / x, self.img_size / y), order=0)
                organ = zoom(organ, (self.img_size / x_o, self.img_size / y_o), order=3)  # why not 3?
        else:
            h5f = h5py.File(self._base_dir + "/ACDC_training_volumes/{}".format(case), 'r')
            image = h5f['image'][:]
            label = h5f['label'][:]

            h5f_o = h5py.File("./dataset/ACDC_SwinUNet_Organ/ACDC_training_volumes/{}".format(case), 'r')
            organ = h5f_o['organ'][:]
        
        sample = {'image': image, 'label': label, 'organ':organ}
        sample["idx"] = idx
        sample['case_name'] = case.replace('.h5', '')
       
        if self.x_transform is not None:
            sample['image'] = self.x_transform(sample['image'].copy())
            sample['organ'] = self.x_transform(sample['organ'].copy())
        if self.y_transform is not None:
            sample['label'] = self.y_transform(sample['label'].copy())

        return sample


def mask_to_onehot(mask, ):
    """
    Converts a segmentation mask (H, W, C) to (H, W, K) where the last dim is a one
    hot encoding vector, C is usually 1 or 3, and K is the number of class.
    """
    semantic_map = []
    mask = np.expand_dims(mask,-1)
    for colour in range (4):
        equality = np.equal(mask, colour)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1).astype(np.int32)
    return semantic_map

def augment_seg(img_aug, img, seg, organ ):
    seg = mask_to_onehot(seg)
    aug_det = img_aug.to_deterministic() 
    image_aug = aug_det.augment_image( img )
    organ_aug = aug_det.augment_image(organ)
    segmap = ia.SegmentationMapOnImage( seg , nb_classes=np.max(seg)+1 , shape=img.shape )
    segmap_aug = aug_det.augment_segmentation_maps( segmap )
    segmap_aug = segmap_aug.get_arr_int()
    segmap_aug = np.argmax(segmap_aug, axis=-1).astype(np.float32)
    return image_aug, segmap_aug, organ_aug

def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label

def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)
