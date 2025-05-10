from __future__ import absolute_import, division, print_function
import albumentations as A
from albumentations.pytorch import ToTensorV2

import math
import os
import random

import PIL.Image as pil
import matplotlib.pyplot as PLT
import cv2

import numpy as np

import torch
import torch.utils.data as data
from scipy.ndimage.filters import gaussian_filter

from torchvision import transforms


used_image_names = set()  # GLOBAL TRACKER
def color_match(rgb_img, target_rgb, tolerance=10):
    return np.all(np.abs(rgb_img - target_rgb) <= tolerance, axis=-1)

def map_to_multiclass(mask_np):
    result = np.zeros(mask_np.shape[:2], dtype=np.uint8)

    result[color_match(mask_np, [255, 255, 255])] = 0  # background
    result[color_match(mask_np, [0, 0, 0])] = 1        # road
    result[color_match(mask_np, [169, 169, 169])] = 2  # lane
    result[color_match(mask_np, [253, 240, 1])] = 3    # vehicle

    return result




def pil_loader(path):
    with open(path, 'rb') as f:
        with pil.open(f) as img:
            return img.convert('RGB')

def process_topview(topview, size):
    topview = topview.convert("1")
    topview = topview.resize((size, size), pil.NEAREST)
    topview = topview.convert("L")
    topview = np.array(topview)
    topview_n = np.zeros(topview.shape)
    topview_n[topview == 255] = 1  # [1.,0.]
    return topview_n

def resize_topview(topview, size):
    topview = topview.convert("1")
    topview = topview.resize((size, size), pil.NEAREST)
    topview = topview.convert("L")
    topview = np.array(topview)
    return topview

def process_discr(topview, size):
    topview = resize_topview(topview, size)
    topview_n = np.zeros((size, size, 2))
    topview_n[topview == 255, 1] = 1.
    topview_n[topview == 0, 0] = 1.
    return topview_n
class MonoDataset(data.Dataset):
    def __init__(self, opt, filenames, is_train=True):
        super(MonoDataset, self).__init__()

        self.opt = opt
        self.data_path = self.opt.data_path
        self.filenames = filenames
        self.is_train = is_train
        self.height = self.opt.height
        self.width = self.opt.width
        self.interp = pil.ANTIALIAS
        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()

        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)

            transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1

        self.resize = transforms.Resize(
            (self.height, self.width), interpolation=self.interp)

    def preprocess(self, inputs, color_aug):

        inputs["color"] = color_aug(self.resize(inputs["color"]))
        for key in inputs.keys():
            if key != "color" and "discr" not in key and key != "filename":
                inputs[key] = process_topview(
                    inputs[key], self.opt.occ_map_size)
            elif key != "filename":
                inputs[key] = self.to_tensor(inputs[key])

    def __len__(self):
        return len(self.filenames)

    def get_color(self, path, do_flip):
        color = self.loader(path)
        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color

    def get_static(self, path, do_flip):
        tv = self.loader(path)

        if do_flip:
            tv = tv.transpose(pil.FLIP_LEFT_RIGHT)

        return tv.convert('L')

    def get_dynamic(self, path, do_flip):
        tv = self.loader(path)

        if do_flip:
            tv = tv.transpose(pil.FLIP_LEFT_RIGHT)

        return tv.convert('L')

    def get_osm(self, path, do_flip):
        osm = self.loader(path)
        return osm

    def get_static_gt(self, path, do_flip):
        tv = self.loader(path)
        return tv.convert('L')

    def get_dynamic_gt(self, path, do_flip):
        tv = self.loader(path)
        return tv.convert('L')


class KITTIObject(MonoDataset):
    """KITTI dataset which loads the original velodyne depth maps for ground truth
    """

    def __init__(self, *args, **kwargs):
        super(KITTIObject, self).__init__(*args, **kwargs)
        self.root_dir = "./data/object"

    def get_image_path(self, root_dir, frame_index):
        image_dir = os.path.join(root_dir, 'image_2')
        img_path = os.path.join(image_dir, "%06d.png" % int(frame_index))
        return img_path

    def get_dynamic_path(self, root_dir, frame_index):
        tv_dir = os.path.join(root_dir, 'vehicle_256')
        tv_path = os.path.join(tv_dir, "%06d.png" % int(frame_index))
        return tv_path

    def get_dynamic_gt_path(self, root_dir, frame_index):
        return self.get_dynamic_path(root_dir, frame_index)

    def get_static_gt_path(self, root_dir, frame_index):
        pass

    def __getitem__(self, index):
        inputs = {}

        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5

        frame_index = self.filenames[index]  # .split()
        # check this part from original code if the dataset is changed
        folder = self.opt.data_path
        inputs["filename"] = frame_index
        inputs["color"] = self.get_color(self.get_image_path(folder, frame_index), do_flip)
        if self.is_train:
            inputs["dynamic"] = self.get_dynamic(
                self.get_dynamic_path(folder, frame_index), do_flip)
        else:
            inputs["dynamic_gt"] = self.get_dynamic_gt(
                self.get_dynamic_gt_path(folder, frame_index), do_flip)


        if do_color_aug:
            color_aug = transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)

        self.preprocess(inputs, color_aug)
        return inputs


class KITTIOdometry(MonoDataset):
    def __init__(self, *args, **kwargs):
        super(KITTIOdometry, self).__init__(*args, **kwargs)
        self.root_dir = "./data/odometry/sequences/"

    def get_image_path(self, root_dir, frame_index):
        file_name = frame_index.replace("road_dense128", "image_2")

        img_path = os.path.join(root_dir, file_name)
        return img_path

    def get_static_path(self, root_dir, frame_index):
        path = os.path.join(root_dir, frame_index)
        return path

    def get_osm_path(self, root_dir):
        osm_file = np.random.choice(os.listdir(root_dir))
        osm_path = os.path.join(root_dir, osm_file)

        return osm_path

    def get_static_gt_path(self, root_dir, frame_index):
        return self.get_static_path(root_dir, frame_index)

    def get_dynamic_gt_path(self, root_dir, frame_index):
        pass

    def __getitem__(self, index):
        inputs = {}

        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5

        frame_index = self.filenames[index]  # .split()
        # check this part from original code if the dataset is changed
        folder = self.opt.data_path
        inputs["filename"] = frame_index
        inputs["color"] = self.get_color(self.get_image_path(folder, frame_index), do_flip)
        if self.is_train:
            inputs["static"] = self.get_static(
                self.get_static_path(folder, frame_index), do_flip)
        else:
            inputs["static_gt"] = self.get_static_gt(
                self.get_static_gt_path(folder, frame_index), do_flip)
        # inputs["osm"] = self.get_osm(folder, frame_index, do_flip)

        if do_color_aug:
            color_aug = transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)

        self.preprocess(inputs, color_aug)
        return inputs


class KITTIRAW(MonoDataset):
    def __init__(self, *args, **kwargs):
        super(KITTIRAW, self).__init__(*args, **kwargs)
        self.root_dir = "./data/raw/"

    def get_image_path(self, root_dir, frame_index):
        img_path = os.path.join(root_dir, frame_index)
        return img_path

    def get_static_path(self, root_dir, frame_index):
        path = os.path.join(
            root_dir, frame_index.replace(
                "image_02/data", "road_256"))
        return path

    def get_osm_path(self, root_dir):
        osm_file = np.random.choice(os.listdir(root_dir))
        osm_path = os.path.join(root_dir, osm_file)

        return osm_path

    def get_static_gt_path(self, root_dir, frame_index):
        path = os.path.join(
            root_dir, frame_index.replace(
                "image_02/data", "road_256"))
        return path

    def get_dynamic_gt_path(self, root_dir, frame_index):
        pass

    def __getitem__(self, index):
        inputs = {}

        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5

        frame_index = self.filenames[index]  # .split()
        # check this part from original code if the dataset is changed
        folder = self.opt.data_path
        inputs["filename"] = frame_index
        inputs["color"] = self.get_color(self.get_image_path(folder, frame_index), do_flip)
        if self.is_train:
            inputs["static"] = self.get_static(
                self.get_static_path(folder, frame_index), do_flip)
        else:
            inputs["static_gt"] = self.get_static_gt(
                self.get_static_gt_path(folder, frame_index), do_flip)
        # inputs["osm"] = self.get_osm(folder, frame_index, do_flip)

        if do_color_aug:
            color_aug = transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)

        self.preprocess(inputs, color_aug)
        return inputs


class Argoverse(MonoDataset):
    def __init__(self, *args, **kwargs):
        super(Argoverse, self).__init__(*args, **kwargs)
        self.root_dir = "./data/argo"

    def get_image_path(self, root_dir, frame_index):
        file_name = frame_index.replace(
            "road_gt", "stereo_front_left").replace(
            "png", "jpg")
        img_path = os.path.join(root_dir, file_name)
        return img_path

    def get_static_path(self, root_dir, frame_index):
        path = os.path.join(root_dir, frame_index)
        return path

    def get_osm_path(self, root_dir):
        osm_file = np.random.choice(os.listdir(root_dir))
        osm_path = os.path.join(root_dir, osm_file)

        return osm_path

    def get_dynamic_path(self, root_dir, frame_index):
        file_name = frame_index.replace(
            "road_gt", "car_bev_gt").replace(
            "png", "jpg")
        path = os.path.join(root_dir, file_name)
        return path

    def get_static_gt_path(self, root_dir, frame_index):
        path = os.path.join(
            root_dir,
            frame_index).replace(
            "road_bev",
            "road_gt")
        return path

    def get_dynamic_gt_path(self, root_dir, frame_index):
        return self.get_dynamic_path(root_dir, frame_index)

    def __getitem__(self, index):
        inputs = {}

        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5

        frame_index = self.filenames[index]  # .split()
        # check this part from original code if the dataset is changed
        folder = self.opt.data_path
        inputs["filename"] = frame_index
        inputs["color"] = self.get_color(self.get_image_path(folder, frame_index), do_flip)

        if self.is_train:
            inputs["dynamic"] = self.get_dynamic(
                self.get_dynamic_path(folder, frame_index), do_flip)
            inputs["static"] = self.get_static(
                self.get_static_path(folder, frame_index), do_flip)
            if self.opt.type == "dynamic":
                inputs["discr"] = process_discr(
                    inputs["dynamic"], self.opt.occ_map_size)
        else:
            if self.opt.type == "dynamic":
                inputs["dynamic_gt"] = self.get_dynamic_gt(
                    self.get_dynamic_gt_path(folder, frame_index), do_flip)
            elif self.opt.type == "static":
                inputs["static_gt"] = self.get_static_gt(
                    self.get_static_gt_path(folder, frame_index), do_flip)

        if do_color_aug:
            color_aug = transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)

        self.preprocess(inputs, color_aug)
        return inputs

class CustomTrafficDataset(data.Dataset):
    def __init__(self, opt, filenames, is_train=True):
        super(CustomTrafficDataset, self).__init__()
        self.opt = opt
        self.data_path = opt.data_path
        self.filenames = filenames
        self.is_train = is_train
        self.height = opt.height
        self.width = opt.width

        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()
        self.resize = transforms.Resize((self.height, self.width), interpolation=pil.BILINEAR)

        # === Split augmentations ===
        self.geometric_aug = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.4),
            A.RandomRotate90(p=0.3),
            A.Resize(height=self.height, width=self.width)
        ], additional_targets={'mask': 'mask'})

        self.color_aug = A.Compose([
            A.GaussNoise(p=0.5),
            A.MotionBlur(p=0.4),
            A.MedianBlur(p=0.5),
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05, p=0.5),
            A.CLAHE(clip_limit=3.0, p=0.5),
            A.RandomGamma(gamma_limit=(90, 110), p=0.5),
        ])
        # Note: Color-only augmentations apply separately only to color image!

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        inputs = {}
        filename = self.filenames[index].split(".")[0]

        color_path = os.path.join(self.data_path, "trainA", filename + ".jpg")
        target_path = os.path.join(self.data_path, "trainB", filename + ".png")

        color_img = self.loader(color_path)
        target_img = self.loader(target_path)

        color_img = color_img.resize((self.width, self.height), pil.BILINEAR)
        target_img = target_img.resize((self.width, self.height), pil.NEAREST)

        color_np = np.array(color_img)
        target_np = np.array(target_img)

        if self.is_train:
            # First apply geometric transformations to both
            augmented = self.geometric_aug(image=color_np, mask=target_np)
            color_np_aug = augmented["image"]
            target_np_aug = augmented["mask"]

            # Then apply color-only transformations to color image only
            color_np_aug = self.color_aug(image=color_np_aug)["image"]
        else:
            color_np_aug = color_np
            target_np_aug = target_np

        inputs["filename"] = filename
        inputs["color"] = pil.fromarray(color_np_aug)

        if target_np_aug.ndim == 2:
            target_np_aug = np.stack([target_np_aug]*3, axis=-1)

        mask_class = map_to_multiclass(target_np_aug)
        mask_class = cv2.resize(mask_class, (self.opt.occ_map_size, self.opt.occ_map_size),
                                interpolation=cv2.INTER_NEAREST)

        if self.is_train:
            inputs["dynamic"] = torch.from_numpy(mask_class).long()
        else:
            inputs["dynamic_gt"] = torch.from_numpy(mask_class).long()

        inputs["color"] = self.to_tensor(self.resize(inputs["color"]))
        return inputs
