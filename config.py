import albumentations as A
from albumentations.pytorch import ToTensorV2

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" # Hack for system issue

import warnings
warnings.filterwarnings("ignore") # Getting bombarded with the same deprecation warning.

ROOT = 'trainval/'
USE_GPU = True
TRAIN_RATIO = 0.8
TRAIN_BATCH_SIZE = 4
GRAD_ACCUM_BATCH = 8
TRAIN_WORKERS = 2
VAL_BATCH_SIZE = 4
VAL_WORKERS = 2
NUM_CLASSES = 2
LR = 1e-4
EARLY_STOP_PATIENCE = 5
PRECISION = 32
EPOCHS = 25
IMG_SIZE = 640

TRANFORMS = A.Compose([
    A.RandomSizedBBoxSafeCrop(width=IMG_SIZE, height=IMG_SIZE),
    A.HorizontalFlip(p=0.3),
    A.OneOf([
            A.Affine(rotate=(-30,30)),
            A.Affine(shear=(-15,15)),
            A.Affine(translate_percent=(0.05,0.15)),
            A.Affine(scale=(0.9,1.1))
    ],p=0.5),
    A.OneOf([
            A.GaussianBlur(blur_limit=(1,5)),
            A.GaussNoise(var_limit=(10,50)),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            A.Cutout(num_holes=IMG_SIZE,max_h_size=1,max_w_size=1), # Simulates dropout(blacking out random pixels)
    ],p=0.5),
    A.OneOf([
            A.Cutout(num_holes=2,max_h_size=8,max_w_size=8), # Blacken out small regions
            A.FancyPCA(),
            A.RandomFog(fog_coef_lower=0.1,fog_coef_upper=0.3),
            A.RandomShadow(),
            A.RandomRain(),
            A.RandomSnow(),
            A.RandomSunFlare(),
    ],p=0.5),
    A.Normalize(mean=(0, 0, 0), std=(1, 1, 1),), # min-max normalization
    ToTensorV2()
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))