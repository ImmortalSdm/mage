import sys
import os
sys.path.append(os.getcwd())

from glob import glob
import logging
import random
import cv2
import torch

import albumentations as A
import numpy as np
import imgaug.augmenters as iaa

from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms as T
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
# from torchvision.datasets import CocoCaptions

from skimage.transform import rescale, resize
from albumentations import DualIAATransform, to_tuple
import torchvision.transforms as transforms
# from evaluate.in_colorization_dataloader import DatasetColorization
# from util.caption_process import CaptionProcessor
# from util.tokenizer import get_tokenizer

# from eval import evaluate_image_captioning  # don't ask me why this import works
logger = logging.getLogger(__name__)

# get images and annotations from https://cocodataset.org/#download
COCO_ROOT_TRAIN = '/mnt/v-dsheng/data/coco_dataset/coco/train2017'
COCO_ROOT_VAL = '/mnt/v-dsheng/data/coco_dataset/coco/val2017'
COCO_ANN_TRAIN = '/mnt/v-dsheng/data/coco_dataset/coco/annotations/captions_train2017.json'
COCO_ANN_VAL   = '/mnt/v-dsheng/data/coco_dataset/coco/annotations/captions_val2017.json'
COCO_P_TRAIN = '/mnt/v-dsheng/data/coco_dataset/coco/panoptic_train2017'
COCO_P_VAL = '/mnt/v-dsheng/data/coco_dataset/coco/panoptic_val2017'
COCO_S_TRAIN = '/mnt/v-dsheng/data/coco_dataset/coco/stuff_164k_seg/train2017'
COCO_S_VAL = '/mnt/v-dsheng/data/coco_dataset/coco/stuff_164k_seg/val2017'
COCO_K_TRAIN = '/mnt/v-dsheng/data/coco_dataset/coco/keypoint_val2017'
COCO_K_VAL = '/mnt/v-dsheng/data/coco_dataset/coco/keypoint_val2017'

'''
Inpainting
'''
class IAAAffine2(DualIAATransform):  # 在输入上放置一个规则的点网格，并通过仿射变换随机移动这些点的邻域。
    """Place a regular grid of points on the input and randomly move the neighbourhood of these point around
    via affine transformations.

    Note: This class introduce interpolation artifacts to mask if it has values other than {0;1}

    Args:
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image, mask
    """

    def __init__(
        self,
        scale=(0.7, 1.3),
        translate_percent=None,
        translate_px=None,
        rotate=0.0,
        shear=(-0.1, 0.1),
        order=1,
        cval=0,
        mode="reflect",
        always_apply=False,
        p=0.5,
    ):
        super(IAAAffine2, self).__init__(always_apply, p)
        self.scale = dict(x=scale, y=scale)
        self.translate_percent = to_tuple(translate_percent, 0)
        self.translate_px = to_tuple(translate_px, 0)
        self.rotate = to_tuple(rotate)
        self.shear = dict(x=shear, y=shear)
        self.order = order
        self.cval = cval
        self.mode = mode

    @property
    def processor(self):
        return iaa.Affine(
            self.scale,
            self.translate_percent,
            self.translate_px,
            self.rotate,
            self.shear,
            self.order,
            self.cval,
            self.mode,
        )

    def get_transform_init_args_names(self):
        return ("scale", "translate_percent", "translate_px", "rotate", "shear", "order", "cval", "mode")


class IAAPerspective2(DualIAATransform):
    """Perform a random four point perspective transform of the input.

    Note: This class introduce interpolation artifacts to mask if it has values other than {0;1}

    Args:
        scale ((float, float): standard deviation of the normal distributions. These are used to sample
            the random distances of the subimage's corners from the full image's corners. Default: (0.05, 0.1).
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image, mask
    """

    def __init__(self, scale=(0.05, 0.1), keep_size=True, always_apply=False, p=0.5,
                 order=1, cval=0, mode="replicate"):
        super(IAAPerspective2, self).__init__(always_apply, p)
        self.scale = to_tuple(scale, 1.0)
        self.keep_size = keep_size
        self.cval = cval
        self.mode = mode

    @property
    def processor(self):
        return iaa.PerspectiveTransform(self.scale, keep_size=self.keep_size, mode=self.mode, cval=self.cval)  # 透视变换

    def get_transform_init_args_names(self):
        return ("scale", "keep_size")


def get_transforms(transform_variant, out_size):
    if transform_variant == 'default':
        transform = A.Compose([
            A.RandomScale(scale_limit=0.2),  # +/- 20%
            A.PadIfNeeded(min_height=out_size, min_width=out_size),
            A.RandomCrop(height=out_size, width=out_size),
            A.HorizontalFlip(),
            A.CLAHE(),  # 限制对比度的自适应直方图均衡：增强图像的对比度的同时可以抑制噪声
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=30, val_shift_limit=5),
            A.ToFloat()
        ])
    elif transform_variant == 'simple':
        transform = A.Compose([
            A.RandomResizedCrop(height=out_size, width=out_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
            A.RandomHorizontalFlip(),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    elif transform_variant == 'distortions':
        transform = A.Compose([
            IAAPerspective2(scale=(0.0, 0.06)),  # 透视变换
            IAAAffine2(scale=(0.7, 1.3),  # 仿射变换
                       rotate=(-40, 40),
                       shear=(-0.1, 0.1)),
            A.PadIfNeeded(min_height=out_size, min_width=out_size), # padding
            A.OpticalDistortion(),   #  Barrel/Pincushion 变形
            A.RandomCrop(height=out_size, width=out_size), 
            A.HorizontalFlip(),  # 翻转
            A.CLAHE(),  # 增强对比度
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),  # 随机改变亮度和对比度
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=30, val_shift_limit=5),  # 随机改变输入图像的色调、饱和度和数值
            A.ToFloat()
        ])
    elif transform_variant == 'distortions_scale05_1':
        transform = A.Compose([
            IAAPerspective2(scale=(0.0, 0.06)),
            IAAAffine2(scale=(0.5, 1.0),
                       rotate=(-40, 40),
                       shear=(-0.1, 0.1),
                       p=1),
            A.PadIfNeeded(min_height=out_size, min_width=out_size),
            A.OpticalDistortion(),
            A.RandomCrop(height=out_size, width=out_size),
            A.HorizontalFlip(),
            A.CLAHE(),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=30, val_shift_limit=5),
            A.ToFloat()
        ])
    elif transform_variant == 'distortions_scale03_12':
        transform = A.Compose([
            IAAPerspective2(scale=(0.0, 0.06)),
            IAAAffine2(scale=(0.3, 1.2),
                       rotate=(-40, 40),
                       shear=(-0.1, 0.1),
                       p=1),
            A.PadIfNeeded(min_height=out_size, min_width=out_size),
            A.OpticalDistortion(),
            A.RandomCrop(height=out_size, width=out_size),
            A.HorizontalFlip(),
            A.CLAHE(),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=30, val_shift_limit=5),
            A.ToFloat()
        ])
    elif transform_variant == 'distortions_scale03_07':
        transform = A.Compose([
            IAAPerspective2(scale=(0.0, 0.06)),
            IAAAffine2(scale=(0.3, 0.7),  # scale 512 to 256 in average
                       rotate=(-40, 40),
                       shear=(-0.1, 0.1),
                       p=1),
            A.PadIfNeeded(min_height=out_size, min_width=out_size),
            A.OpticalDistortion(),
            A.RandomCrop(height=out_size, width=out_size),
            A.HorizontalFlip(),
            A.CLAHE(),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=30, val_shift_limit=5),
            A.ToFloat()
        ])
    elif transform_variant == 'distortions_light':
        transform = A.Compose([
            IAAPerspective2(scale=(0.0, 0.02)),
            IAAAffine2(scale=(0.8, 1.8),
                       rotate=(-20, 20),
                       shear=(-0.03, 0.03)),
            A.PadIfNeeded(min_height=out_size, min_width=out_size),
            A.RandomCrop(height=out_size, width=out_size),
            A.HorizontalFlip(),
            A.CLAHE(),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=30, val_shift_limit=5),
            A.ToFloat()
        ])
    elif transform_variant == 'non_space_transform':
        transform = A.Compose([
            A.CLAHE(),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=30, val_shift_limit=5),
            A.ToFloat()
        ])
    elif transform_variant == 'no_augs':
        transform = A.Compose([
            A.ToFloat()
        ])
    else:
        raise ValueError(f'Unexpected transform_variant {transform_variant}')
    return transform

class MyInpaintingTrainDataset(Dataset):  # My
    def __init__(self, data_dir, transform, num_samples, input_size=128):
        # imgs = sorted(list(glob(os.path.join(data_dir, 'image/*'), recursive=True)))
        # masks = sorted(list(glob(os.path.join(data_dir, 'mask/*'), recursive=True)))
        imgs = sorted(list(glob(os.path.join(data_dir, '*'), recursive=True)))
        masks = sorted(list(glob(os.path.join(data_dir, '*_mask.png'), recursive=True)))

        self.in_files = list(set(imgs).difference(set(masks)))
        self.in_masks = masks
        assert len(self.in_files) == len(self.in_masks)
        # print(len(self.in_files), len(self.in_masks))
        del masks, imgs

        self.size = input_size
        self.num_samples = num_samples
        self.transform = transform
        self.iter_i = 0

    def __len__(self):
        return len(self.in_files)

    def __getitem__(self, item):
        img_path = self.in_files[item]
        mask_path = self.in_masks[item]
        img = np.array(Image.open(img_path).convert('RGB'))
        img = cv2.resize(img, (self.size, self.size))
        mask = np.array(Image.open(mask_path).convert('L'))
        mask = cv2.resize(mask, (self.size, self.size))
        mask = (mask>10).astype('uint8')

        # mask = np.ones((self.size, self.size))
        # x = random.randint(0, self.size-64)
        # y = random.randint(0, self.size-64)
        # mask[x:x+64, y:y+64] = 0
        mask = mask[..., None]
        masked_img = (1-mask)*img

        if self.transform:
            masked_img = self.transform(Image.fromarray(np.uint8(masked_img)))
            img = self.transform(Image.fromarray(np.uint8(img)))
        
        samples = []
        for i in range(self.num_samples):
            sample_img = np.array(Image.open(random.choice(self.in_files)).convert('RGB'))
            # print(sample_img.shape)
            sample_img = cv2.resize(sample_img, (self.size, self.size))
            sample_mask = np.array(Image.open(random.choice(self.in_masks)).convert('L'))
            sample_mask = cv2.resize(sample_mask, (self.size, self.size))
            sample_mask = (sample_mask>10).astype('uint8')[..., None]
            masked_sample_img = (1-sample_mask)*sample_img

            if self.transform:
                masked_sample_img = self.transform(Image.fromarray(np.uint8(masked_sample_img)))
                sample_img = self.transform(Image.fromarray(np.uint8(sample_img)))
            samples.append(masked_sample_img)
            samples.append(sample_img)
        if self.transform:
            samples = torch.stack(samples, dim=0)
        else:
            samples = np.stack(samples, axis=0)
            
        # sample_img = cv2.imread(sample_path)
        # sample_img = resize(sample_img, self.size)
        # sample_img = cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB)
        # sample_img = self.transform(image=sample_img)['image']
        # sample_img = np.transpose(sample_img, (2, 0, 1)) 
        # print(img, mask)
        # cv2.imshow("Image", mask)
        # cv2.waitKey (0)
        self.iter_i += 1
        return dict(m_img=masked_img,
                    img=img,
                    samples=samples
                    )

'''
Caption
'''
from torchvision.datasets.vision import VisionDataset

class CocoCaptions(VisionDataset):
    def __init__(
        self,
        root,
        annFile,
        transform=None,
        target_transform=None,
        transforms=None,
    ):
        super().__init__(root, transforms, transform, target_transform)
        from pycocotools.coco import COCO
        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.anns.keys()))

    def __getitem__(self, index):
        record = self.coco.anns[self.ids[index]]
        target = record["caption"]
        img_id = record["image_id"]

        path = self.coco.loadImgs(img_id)[0]["file_name"]
        image = Image.open(os.path.join(self.root, path)).convert("RGB")

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self):
        return len(self.ids)

def prepare_training_dataset(transform):
    """ prepare a CocoCaptions training dataset """

    return CocoCaptions(
        COCO_ROOT_TRAIN, 
        COCO_ANN_TRAIN, 
        transform=transform
    )
    
def prepare_evaluation_dataset(transform):
    return CocoCaptions(COCO_ROOT_VAL, COCO_ANN_VAL, 
        transform=transform)



class MyCOCOCaptionDataset(Dataset):  # My
    def __init__(self, data_type, data_list, transform): 
        if data_type == 'train':
            self.ds = prepare_training_dataset(transform)
        else:
            self.ds = prepare_evaluation_dataset(transform)
        # self.tokenizer = get_tokenizer()
        self.iter_i = 0

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, item):
        sample_pair = self.ds[item]
        refer_pair = random.choice(self.ds)

        self.iter_i += 1
        return dict(sample_pair=sample_pair,
                    refer_pair=refer_pair
                    )

class DataCollator:
    def __init__(self, config):
        # self.processor = CaptionProcessor(config)
        self.processor = get_tokenizer()
        
    def __call__(self, batch):
        pixel_values, sentences = zip(*batch)
        inputs = self.processor(text=sentences)
        pixel_values = torch.stack(pixel_values)
        
        return dict(
            pixel_values=pixel_values,
            labels=inputs['input_ids'],
            **inputs
        )

'''
visual_prompt_dataset
'''
class MyVisualPromptDataset(Dataset):  # My
    def __init__(self, data_list, transform, mask_transform): 
        with open(data_list, "r") as f:
            img_train = []
            for line in f.readlines():
                line = line.strip('\n')  #去掉列表中每一个元素的换行符
                line = line.split(' ')  #去掉列表中每一个元素的换行符
                img_train.append(line)
        self.ds = img_train
        self.transoform = transform
        self.mask_transform = mask_transform
        # self.tokenizer = get_tokenizer()
        self.iter_i = 0
        del img_train

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        # sample_refer_path, sample_gt_path, refer_path, gt_path = self.ds[idx][0], self.ds[idx][1], self.ds[idx][2], self.ds[idx][3]
        name, sample_refer_path, sample_gt_path, refer_path, gt_path = self.ds[idx][0], self.ds[idx][1], self.ds[idx][2], self.ds[idx][3], self.ds[idx][4]
        sample_refer, sample_gt, refer, gt = Image.open(sample_refer_path).convert('RGB'), Image.open(sample_gt_path).convert('RGB'), Image.open(refer_path).convert('RGB'), Image.open(gt_path).convert('RGB')
        if sample_refer_path.split('/')[-3] == 'ILSVRC_2012':
            sample_refer, sample_gt = self.mask_transform(sample_refer), self.transoform(sample_gt)
            refer, gt = self.mask_transform(refer), self.transoform(gt)
        else:
            sample_refer, sample_gt = self.transoform(sample_refer), self.transoform(sample_gt)            
            refer, gt = self.transoform(refer), self.transoform(gt)

        self.iter_i += 1
        return dict(sample_refer=sample_refer,
                    sample_gt=sample_gt,
                    refer=refer,
                    gt=gt,
                    name=name
                    )

def get_dataset_by_type(data_type, data_path=None, transforms=None, input_size=128, samples_num=None):
    train_path = os.path.join(data_path, 'train')
    train_list = os.path.join(data_path, 'imagenetcolor_train.txt')
    val_path = os.path.join(data_path, 'val')
    val_list = os.path.join(data_path, 'imagenetcolor_val.txt')
    if data_type == 'CVF':
        dataset_train = datasets.ImageFolder(train_path, transform=transforms)
        dataset_val = datasets.ImageFolder(val_path, transform=transforms)
    elif data_type == 'color':
        image_transform = transforms.Compose([
            transforms.CenterCrop((224 // 2, 224 // 2)),
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor()])
        mask_transform = transforms.Compose([
            transforms.CenterCrop((224 // 2, 224 // 2)),
            transforms.Resize((input_size, input_size)),
            transforms.Grayscale(3),
            transforms.ToTensor()])
        dataset_train = DatasetColorization(train_list, train_path, image_transform, mask_transform, padding=0)
        dataset_val = DatasetColorization(val_list, val_path, image_transform, mask_transform, padding=0)
    elif data_type == 'caption':
        dataset_train = MyCOCOCaptionDataset('train', None, transform)
        dataset_val = MyCOCOCaptionDataset('val', None, transform)
    else:
        dataset_train = MyInpaintingTrainDataset(train_path, transforms, samples_num)
        dataset_val = MyInpaintingTrainDataset(val_path, transforms, samples_num)

'''
Colorization
'''
class DatasetColorization(Dataset):
    def __init__(self, file_list, datapath, image_transform, mask_transform, num_samples=100, padding: bool = 1,
                 use_original_imgsize: bool = False, flipped_order: bool = False,
                 reverse_support_and_query: bool = False, random: bool = False):
        self.padding = padding
        self.random = random
        self.use_original_imgsize = use_original_imgsize
        self.num_samples = num_samples
        self.image_transform = image_transform
        self.reverse_support_and_query = reverse_support_and_query
        self.mask_transform = mask_transform
        with open(file_list, "r") as f:
            imgs = []
            for line in f.readlines():
                line = line.strip('\n')
                line = line.split(' ')
                imgs.append(line)
        self.data_dir = datapath
        self.ds = imgs
        # self.ds = ImageFolder(datapath)
        self.flipped_order = flipped_order
        del imgs
        # self.indices = np.random.choice(np.arange(0, len(self.ds)-1), size=self.num_samples, replace=False)

    def __len__(self):
        return len(self.ds)

    def create_grid_from_images(self, support_img, support_mask, query_img, query_mask):
        if self.reverse_support_and_query:
            support_img, support_mask, query_img, query_mask = query_img, query_mask, support_img, support_mask
        canvas = torch.ones((support_img.shape[0], 2 * support_img.shape[1] + 2 * self.padding,
                             2 * support_img.shape[2] + 2 * self.padding))
        canvas[:, :support_img.shape[1], :support_img.shape[2]] = support_img
        if self.flipped_order:
            canvas[:, :support_img.shape[1], -support_img.shape[2]:] = query_img
            canvas[:, -query_img.shape[1]:, -support_img.shape[2]:] = query_mask
            canvas[:, -query_img.shape[1]:, :query_img.shape[2]] = support_mask
        else:
            canvas[:, -query_img.shape[1]:, :query_img.shape[2]] = query_img
            canvas[:, :support_img.shape[1], -support_img.shape[2]:] = support_mask
            canvas[:, -query_img.shape[1]:, -support_img.shape[2]:] = query_mask

        return canvas

    def __getitem__(self, idx):
        # support_idx = np.random.choice(np.arange(0, len(self)-1))
        # idx = self.indices[idx]
        query_path, support_path = os.path.join(self.data_dir, self.ds[idx][0]), os.path.join(self.data_dir, self.ds[idx][1])
        query, support = Image.open(query_path).convert('RGB'), Image.open(support_path).convert('RGB')
        query_mask, query_img = self.mask_transform(query), self.image_transform(query)
        support_mask, support_img = self.mask_transform(support), self.image_transform(support)
        # grid = self.create_grid_from_images(support_img, support_mask, query_img, query_mask)
        batch = {'query_img': query_img, 'query_mask': query_mask, 'support_img': support_img,
                 'support_mask': support_mask} # , 'grid': grid

        return batch

def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    root = os.path.join(args.data_path, 'train' if is_train else 'val')
    dataset = datasets.ImageFolder(root, transform=transform)

    return dataset

def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # mean = (0, 0, 0)
    # std = (1, 1, 1)
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            scale=(0.2, 1.0),
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    size = 292
    t.append(
        transforms.Resize(size, interpolation=Image.BILINEAR if args.interpolation == 'bilinear' else
                          Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)

if __name__=='__main__':
    # data_path = '/mnt/v-dsheng/data/inpainting/places_256/val'
    # process = transforms.Compose([
    #     # transforms.RandomResizedCrop(256, scale=(0.2, 1.0), interpolation=3),
    #     transforms.Resize((256, 256)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    # dataset = MyInpaintingTrainDataset(data_dir=data_path, transform=process, num_samples=1)
    # for data in iter(dataset):
    #     print(type(data['m_img']))
    #     # Image.fromarray(data['m_img']).save('test.png')
    #     print(data['samples'].shape)
    #     # Image.fromarray(data['samples'][0]).save('test_sample.png')

    '''
    caption
    '''
    transform = transforms.Compose([
        transforms.Resize((384, 640)),
        transforms.ToTensor()])
    # eval_dataset = prepare_training_dataset(transform)

    # indices = np.random.choice(np.arange(0, len(eval_dataset)), size=len(eval_dataset), replace=False)
    # support_indices = np.random.choice(np.arange(0, len(eval_dataset)), size=len(eval_dataset), replace=False)

    # img_dir = '/mnt/v-dsheng/data/coco_dataset/coco/train2017'
    # save_path = '/mnt/v-dsheng/data/coco_dataset/coco/visual_prompt_train2017.txt'

    # with open(save_path,"w") as f:
    #     for i in range(len(eval_dataset)):
    #         support_ind = support_indices[i]
    #         ind = indices[i]
    #         f.write(str(support_ind) + ' ')  # 自带文件关闭功能，不需要再写f.close()
    #         f.write(str(ind) + '\n')  # 自带文件关闭功能，不需要再写f.close()
    
    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
    
    setup_seed(233)

    train_path = '/mnt/v-dsheng/data/visual_prompt_dataset/train.txt'
    val_path = '/mnt/v-dsheng/data/visual_prompt_dataset/val.txt'
    image_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()])
    mask_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.Grayscale(3),
        transforms.ToTensor()])
    # dataset = MyVisualPromptDataset(val_path, image_transform, mask_transform)
    # data_loader = torch.utils.data.DataLoader(
    #     dataset,
    #     batch_size=64,
    #     num_workers=8,
    #     pin_memory=True,
    #     drop_last=True,
    # )

    # for i, data in enumerate(data_loader):
    #     print(data['sample_gt'].shape)

    # dataset = MyCOCOCaptionDataset('train', None, transform)
    # print(len(dataset))
    # data_loader = torch.utils.data.DataLoader(
    #     dataset,
    #     batch_size=4,
    #     num_workers=4,
    #     pin_memory=True,
    #     drop_last=True,
    # )

    # for i, data in enumerate(data_loader):
    #     print(data['sample_pair'][0].shape)
    #     print(data['sample_pair'][1])
    #     print(np.array(data['sample_pair'][1]).shape)
