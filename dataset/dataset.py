import os
import cv2
import torch
import random
import numpy as np
import random

from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class MyDataset(Dataset):

    def __init__(self, img_dir, mode):

        self.img_path_list = []
        self.lms_path_list = []
        self.mode = mode

        for i in range(len(os.listdir(img_dir + "/full_body_img/"))):
            img_path = os.path.join(
                img_dir + "/full_body_img/", str(i) + ".jpg")
            lms_path = os.path.join(img_dir + "/landmarks/", str(i) + ".lms")
            self.img_path_list.append(img_path)
            self.lms_path_list.append(lms_path)

        if self.mode == "wenet":
            self.audio_feats = np.load(img_dir + "/aud_wenet.npy")
        if self.mode == "hubert":
            self.audio_feats = np.load(img_dir + "/aud_hu.npy")

        self.audio_feats = self.audio_feats.astype(np.float32)

    def __len__(self):
        # return len(self.img_path_list)-1
        # return len(self.img_path_list)
        return self.audio_feats.shape[0] - 1

    def get_audio_features(self, features, index):
        left = index - 8
        right = index + 8
        pad_left = 0
        pad_right = 0
        if left < 0:
            pad_left = -left
            left = 0
        if right > features.shape[0]:
            pad_right = right - features.shape[0]
            right = features.shape[0]
        auds = torch.from_numpy(features[left:right])
        if pad_left > 0:
            auds = torch.cat([torch.zeros_like(auds[:pad_left]), auds], dim=0)
        if pad_right > 0:
            auds = torch.cat([auds, torch.zeros_like(
                auds[:pad_right])], dim=0)  # [8, 16]
        return auds

    def get_audio_features_1(self, features, index):

        left = index - 8
        pad_left = 0
        if left < 0:
            pad_left = -left
            left = 0
        auds = features[left:index]
        auds = torch.from_numpy(auds)
        if pad_left > 0:
            # pad may be longer than auds, so do not use zeros_like
            auds = torch.cat([torch.zeros(pad_left, *auds.shape[1:], device=auds.device, dtype=auds.dtype), auds],
                             dim=0)
        return auds

    def process_img(self, img, lms_path, img_ex, lms_path_ex):

        lms_list = []
        with open(lms_path, "r") as f:
            lines = f.read().splitlines()
            for line in lines:
                arr = line.split(" ")
                arr = np.array(arr, dtype=np.float32)
                lms_list.append(arr)
        lms = np.array(lms_list, dtype=np.int32)
        xmin = lms[1][0]
        ymin = lms[52][1]

        xmax = lms[31][0]
        width = xmax - xmin
        ymax = ymin + width
        # 裁剪出嘴
        crop_img = img[ymin:ymax, xmin:xmax]
        crop_img = cv2.resize(crop_img, (168, 168), cv2.INTER_AREA)
        # 裁剪出核心区域
        img_real = crop_img[4:164, 4:164].copy()
        # 嘴的roi区域
        img_real_ori = img_real.copy()
        # 得到区域掩码

        img_masked = cv2.rectangle(img_real, (5, 5, 150, 145), (0, 0, 0), -1)
        # img_masked = cv2.rectangle(img_real, (5, 5, 145, 145), (0, 0, 0), -1)
        # img_masked = cv2.rectangle(img_real, (4, 4, 164, 164), (0, 0, 0), -1)

        lms_list = []
        with open(lms_path_ex, "r") as f:
            lines = f.read().splitlines()
            for line in lines:
                arr = line.split(" ")
                arr = np.array(arr, dtype=np.float32)
                lms_list.append(arr)
        lms = np.array(lms_list, dtype=np.int32)
        xmin = lms[1][0]
        ymin = lms[52][1]

        xmax = lms[31][0]
        width = xmax - xmin
        ymax = ymin + width
        # 随机采样一张图片
        crop_img = img_ex[ymin:ymax, xmin:xmax]
        crop_img = cv2.resize(crop_img, (168, 168), cv2.INTER_AREA)
        # 采用的嘴型区域
        img_real_ex = crop_img[4:164, 4:164].copy()

        # 原图像roi
        img_real_ori = img_real_ori.transpose(2, 0, 1).astype(np.float32)
        # 原图像掩码
        img_masked = img_masked.transpose(2, 0, 1).astype(np.float32)
        # 提取出来的嘴型区域
        img_real_ex = img_real_ex.transpose(2, 0, 1).astype(np.float32)

        img_real_ex_T = torch.from_numpy(img_real_ex / 255.0)
        img_real_T = torch.from_numpy(img_real_ori / 255.0)
        img_masked_T = torch.from_numpy(img_masked / 255.0)
        img_concat_T = torch.cat([img_real_ex_T, img_masked_T], axis=0)

        return img_concat_T, img_real_T, img_real_ex_T, img_masked_T

    def __getitem__(self, idx):
        idx = idx if idx <= len(self.img_path_list) - 1 else len(self.img_path_list) - 1
        img = cv2.imread(self.img_path_list[idx])
        lms_path = self.lms_path_list[idx]

        # 创建一个排除了 idx 的索引列表，并从中随机选择一个
        available_indices = np.setdiff1d(np.arange(0, self.__len__()), idx)
        ex_int = np.random.choice(available_indices)
        ex_int = ex_int if ex_int <= len(self.img_path_list) - 1 else len(self.img_path_list) - 1
        img_ex = cv2.imread(self.img_path_list[ex_int])
        lms_path_ex = self.lms_path_list[ex_int]

        img_concat_T, img_real_T, img_real_ex_T, img_masked_T = self.process_img(img, lms_path, img_ex, lms_path_ex)

        # Save images to disk
        # save_dir = 'saved_images/'
        # if not os.path.exists(save_dir):
        #     os.makedirs(save_dir)

        # # Convert tensors to numpy arrays and save them as images
        # img_real_T_np = img_real_T.numpy().transpose(1, 2, 0) * 255  # Real image (real lips)
        # img_masked_T_np = img_masked_T.numpy().transpose(1, 2, 0) * 255  # Masked image (masked lips)

        # # Ensure the images are in [H, W, C] format (height, width, channels)
        # img_real_T_np = np.clip(img_real_T_np, 0, 255).astype(np.uint8)
        # img_masked_T_np = np.clip(img_masked_T_np, 0, 255).astype(np.uint8)

        # # Generate file names and save images
        # img_real_T_filename = os.path.join(save_dir, f"img_real_T_{idx}.png")
        # img_masked_T_filename = os.path.join(save_dir, f"img_masked_T_{idx}.png")

        # # Save the images
        # cv2.imwrite(img_real_T_filename, img_real_T_np)
        # cv2.imwrite(img_masked_T_filename, img_masked_T_np)

        # Continue with audio feature processing...
        audio_feat = self.get_audio_features(self.audio_feats, idx)
        if self.mode == "wenet":
            audio_feat = audio_feat.reshape(256, 16, 32)
        if self.mode == "hubert":
            audio_feat = audio_feat.reshape(32, 32, 32)

        return img_concat_T, img_real_T, audio_feat
