from albumentations.pytorch.transforms import ToTensorV2
import albumentations as A
import cv2
from configs.load_config import get_config

cfg = get_config()

val_transform = A.Compose([
    A.Resize(cfg['init_parm']['input_w'],cfg['init_parm']['input_h']),
    # A.Equalize(p=1.0),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
    ToTensorV2(p=1.0),
])

train_transform = A.Compose([
    A.Resize(cfg['init_parm']['input_w'],cfg['init_parm']['input_h']),
    A.VerticalFlip(p=0.3),
    A.HorizontalFlip(p=0.5),
    A.OneOf([
        A.RandomGridShuffle(grid=(3, 3), always_apply=False, p=0.3),
        A.GridDistortion(num_steps=10, distort_limit=0.5, border_mode=4, always_apply=False, p=0.15),
        A.ElasticTransform(p=0.3, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
        A.OpticalDistortion(p=0.3, distort_limit=2, shift_limit=0.5),
    ], p=0.3),
    A.OneOf([
        A.RandomGamma(gamma_limit=(20, 20), eps=None, always_apply=False, p=0.5),
        A.ChannelDropout(channel_drop_range=(1, 1), fill_value=0, p=0.2),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, always_apply=False, p=0.3),
        A.ColorJitter(p=0.5, brightness=(0.8, 1.2), contrast=(0.8, 1.2), saturation=(0.8, 1.2), hue=(-0.2, 0.2)),
        A.FancyPCA(always_apply=False, p=0.5, alpha=1.78)
    ], p = 0.2),
    A.OneOf([
        A.GaussNoise(p=0.5),    # 将高斯噪声应用于输入图像。
        A.CLAHE(clip_limit=5,tile_grid_size=(8, 8),p=0.5),
    ], p=0.2),   # 应用选定变换的概率
    A.OneOf([
        A.MotionBlur(p=0.3),   # 使用随机大小的内核将运动模糊应用于输入图像。
        A.MedianBlur(blur_limit=3, p=0.3),    # 中值滤波
        A.Blur(blur_limit=3, p=0.3),   # 使用随机大小的内核模糊输入图像。
        A.MultiplicativeNoise(always_apply=False, p=0.3, multiplier=(1.58, 2.13), per_channel=True, elementwise=True)
    ], p=0.3),
    A.Emboss(alpha=(0.2, 0.5), strength=(0.2, 0.7),p=0.1),
    A.Equalize(p=0.3),
    A.OneOf([
        A.ChannelShuffle(always_apply=False, p=0.5),
        A.RGBShift(r_shift_limit=10, g_shift_limit=20, b_shift_limit=20, always_apply=False, p=0.5),
        A.InvertImg(always_apply=False, p=0.5)
    ], p=0.5),
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.3, rotate_limit=45, p=0.3),
    A.RandomBrightnessContrast(p=0.2),   # 随机明亮对比度
    A.CoarseDropout(p=0.3),
    A.MultiplicativeNoise(always_apply=False, p=0.15, multiplier=(1.58, 2.13), per_channel=True, elementwise=True),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1),
    ToTensorV2(p=1.0),
])
# train_transform = A.Compose([
#     A.Resize(cfg['init_parm']['input_w'],cfg['init_parm']['input_h']),
#     A.VerticalFlip(p=0.3),
#     A.GridDistortion(num_steps=10, distort_limit=0.5,border_mode=4, always_apply=False, p=0.15),
#     A.HorizontalFlip(p=0.5),
#     A.OneOf([
#         A.ElasticTransform(p=0.3, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
#         A.OpticalDistortion(p=0.3, distort_limit=2, shift_limit=0.5),
#         A.RandomGamma(gamma_limit=(20, 20), eps=None, always_apply=False, p=0.5)
#     ], p = 0.2),
#     A.OneOf([
#         A.GaussNoise(p=0.5),    # 将高斯噪声应用于输入图像。
#         A.CLAHE(clip_limit=5,tile_grid_size=(8, 8),p=0.5),
#     ], p=0.2),   # 应用选定变换的概率
#     A.OneOf([
#         A.MotionBlur(p=0.3),   # 使用随机大小的内核将运动模糊应用于输入图像。
#         A.MedianBlur(blur_limit=3, p=0.3),    # 中值滤波
#         A.Blur(blur_limit=3, p=0.3),   # 使用随机大小的内核模糊输入图像。
#     ], p=0.2),
#     A.ChannelDropout(channel_drop_range=(1,1),fill_value=0,p=0.2),
#     A.Downscale(scale_min=0.25,scale_max=0.25,interpolation=cv2.INTER_NEAREST,p=0.2),
#     A.Emboss(alpha=(0.2, 0.5), strength=(0.2, 0.7),p=0.3),
#     A.Equalize(p=0.3),
#     A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.3),
#     A.RandomBrightnessContrast(p=0.2),   # 随机明亮对比度
#     A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1),
#     ToTensorV2(p=1.0),
# ])
# train_transform = A.Compose([
#     A.Resize(cfg['init_parm']['input_w'],cfg['init_parm']['input_h']),
#     A.VerticalFlip(p=0.3),
#     A.GridDistortion(num_steps=10, distort_limit=0.3,border_mode=4, always_apply=False, p=0.15),
#     A.HorizontalFlip(p=0.5),
#     A.OneOf([
#         A.ElasticTransform(p=1, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
#         A.OpticalDistortion(p=1, distort_limit=2, shift_limit=0.5),
#         A.RandomGamma(gamma_limit=(20, 20), eps=None, always_apply=False, p=0.5)
#     ], p = 0.2),
#     A.OneOf([
#         A.GaussNoise(),    # 将高斯噪声应用于输入图像。
#         A.CLAHE(clip_limit=5,tile_grid_size=(8, 8),p=1),
#     ], p=0.2),   # 应用选定变换的概率
#     A.OneOf([
#         A.MotionBlur(p=0.2),   # 使用随机大小的内核将运动模糊应用于输入图像。
#         A.Blur(blur_limit=15, always_apply=False, p=0.3),
#         A.MedianBlur(blur_limit=3, p=0.1),    # 中值滤波
#         A.Blur(blur_limit=3, p=0.1),   # 使用随机大小的内核模糊输入图像。
#     ], p=0.2),
#     A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.3),
#     A.RandomBrightnessContrast(p=0.2),   # 随机明亮对比度
#     A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
#     ToTensorV2(p=1.0),
# ])
