import time

import torch
from torch import nn
from torch.nn import functional as F, BatchNorm2d, ReLU, Linear, AdaptiveAvgPool2d
from torch.utils.data import DataLoader
import cv2
import os
import numpy as np
from torch import optim
import random
import argparse


class Dataset(object):
    def __init__(self, dataset_dir, mode):

        self.img_path_list = []
        self.lms_path_list = []

        for i in range(len(os.listdir(dataset_dir + "/full_body_img/"))):
            img_path = os.path.join(
                dataset_dir + "/full_body_img/", str(i) + ".jpg")
            lms_path = os.path.join(
                dataset_dir + "/landmarks/", str(i) + ".lms")
            self.img_path_list.append(img_path)
            self.lms_path_list.append(lms_path)

        if mode == "wenet":
            audio_feats_path = dataset_dir + "/aud_wenet.npy"
        if mode == "hubert":
            audio_feats_path = dataset_dir + "/aud_hu.npy"
        self.mode = mode
        self.audio_feats = np.load(audio_feats_path)
        self.audio_feats = self.audio_feats.astype(np.float32)

    def __len__(self):
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

        crop_img = img[ymin:ymax, xmin:xmax]
        crop_img = cv2.resize(crop_img, (168, 168), cv2.INTER_AREA)
        img_real = crop_img[4:164, 4:164].copy()
        img_real_ori = img_real.copy()
        img_real_ori = img_real_ori.transpose(2, 0, 1).astype(np.float32)
        img_real_T = torch.from_numpy(img_real_ori / 255.0)

        return img_real_T

    def __getitem__(self, idx):
        idx = idx if idx <= len(self.img_path_list) - \
            1 else len(self.img_path_list) - 1
        img = cv2.imread(self.img_path_list[idx])
        lms_path = self.lms_path_list[idx]
        # # 创建一个排除了 idx 的索引列表，并从中随机选择一个
        # available_indices = [i for i in range(len(self.img_path_list)) if i != idx]
        # ex_int = random.choice(available_indices)
        # ex_int = ex_int if ex_int <= len(self.img_path_list) - 1 else len(self.img_path_list) - 1
        # img_ex = cv2.imread(self.img_path_list[ex_int])
        # lms_path_ex = self.lms_path_list[ex_int]
        img_real_T = self.process_img(img, lms_path, 0, 0)
        audio_feat = self.get_audio_features(self.audio_feats, idx)  #
        # print(audio_feat.shape)
        if self.mode == "wenet":
            audio_feat = audio_feat.reshape(256, 16, 32)
        if self.mode == "hubert":
            audio_feat = audio_feat.reshape(32, 32, 32)
        y = torch.ones(1).float()
        t = torch.tensor(idx)

        return img_real_T, audio_feat, y, t


class Conv2d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size, stride, padding),
            nn.BatchNorm2d(cout)
        )
        self.act = nn.ReLU()
        self.residual = residual

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            out += x
        return self.act(out)


class nonorm_Conv2d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size, stride, padding),
        )
        self.act = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x):
        out = self.conv_block(x)
        return self.act(out)


class Conv2dTranspose(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, output_padding=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
            nn.ConvTranspose2d(cin, cout, kernel_size,
                               stride, padding, output_padding),
            nn.BatchNorm2d(cout)
        )
        self.act = nn.ReLU()

    def forward(self, x):
        out = self.conv_block(x)
        return self.act(out)


class SyncNet_color(nn.Module):
    def __init__(self, mode):
        super(SyncNet_color, self).__init__()

        self.face_encoder = nn.Sequential(
            Conv2d(3, 32, kernel_size=(7, 7), stride=1, padding=3),

            Conv2d(32, 64, kernel_size=5, stride=2, padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1,
                   padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1,
                   padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1,
                   padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1,
                   padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1,
                   padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            Conv2d(512, 512, kernel_size=3, stride=1,
                   padding=1, residual=True),
            Conv2d(512, 512, kernel_size=3, stride=1,
                   padding=1, residual=True),

            # Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),
            # Conv2d(1024, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
        )

        p1 = 256
        p2 = (1, 2)
        if mode == "hubert":
            p1 = 32
            p2 = (2, 2)

        self.audio_encoder = nn.Sequential(
            Conv2d(p1, 256, kernel_size=3, stride=1, padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1,
                   padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1,
                   padding=1, residual=True),

            Conv2d(256, 256, kernel_size=3, stride=p2, padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1,
                   padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1,
                   padding=1, residual=True),

            Conv2d(256, 256, kernel_size=3, stride=2, padding=2),
            Conv2d(256, 256, kernel_size=3, stride=1,
                   padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1,
                   padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            Conv2d(512, 512, kernel_size=3, stride=1,
                   padding=1, residual=True),
            Conv2d(512, 512, kernel_size=3, stride=1,
                   padding=1, residual=True),

            # Conv2d(512, 1024, kernel_size=3, stride=1, padding=0),
            # Conv2d(1024, 512, kernel_size=1, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),

        )

        self.audio_leaky_rule = nn.LeakyReLU()
        self.face_leaky_rule = nn.LeakyReLU()

    def forward(self, face_sequences, audio_sequences):  # audio_sequences := (B, dim, T)
        face_embedding = self.face_encoder(face_sequences)
        audio_embedding = self.audio_encoder(audio_sequences)

        audio_embedding = audio_embedding.view(audio_embedding.size(0), -1)
        face_embedding = face_embedding.view(face_embedding.size(0), -1)

        audio_embedding = F.normalize(audio_embedding, p=2, dim=1)
        audio_embedding = self.audio_leaky_rule(audio_embedding)
        face_embedding = F.normalize(face_embedding, p=2, dim=1)
        face_embedding = self.face_leaky_rule(face_embedding)

        return audio_embedding, face_embedding


# class SyncNet_color(nn.Module):
#     def __init__(self, mode):
#         super(SyncNet_color, self).__init__()
#
#         # Define a helper function to add Conv2d, BatchNorm2d and ReLU layers
#         def conv_bn_relu(in_channels, out_channels, kernel_size, stride=1, padding=0, residual=False):
#             layers = [
#                 nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
#                 nn.BatchNorm2d(out_channels),
#                 nn.ReLU()  # Removed inplace=True
#             ]
#             if residual:
#                 layers.append(ResidualBlock(out_channels))
#             return nn.Sequential(*layers)
#
#         # Define a residual block for convenience
#         class ResidualBlock(nn.Module):
#             def __init__(self, channels):
#                 super(ResidualBlock, self).__init__()
#                 self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
#                 self.bn1 = nn.BatchNorm2d(channels)
#                 self.relu1 = nn.ReLU()
#                 self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
#                 self.bn2 = nn.BatchNorm2d(channels)
#                 self.relu2 = nn.ReLU()
#
#             def forward(self, x):
#                 residual = x
#                 x = self.conv1(x)
#                 x = self.bn1(x)
#                 x = self.relu1(x)
#                 x = self.conv2(x)
#                 x = self.bn2(x)
#                 x += residual
#                 return self.relu2(x)  # Apply ReLU after adding the residual
#
#         # Face Encoder
#         self.face_encoder = nn.Sequential(
#             conv_bn_relu(3, 32, kernel_size=(7, 7), stride=1, padding=3),
#
#             conv_bn_relu(32, 64, kernel_size=5, stride=2, padding=1),
#             conv_bn_relu(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
#             conv_bn_relu(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
#
#             conv_bn_relu(64, 128, kernel_size=3, stride=2, padding=1),
#             conv_bn_relu(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
#             conv_bn_relu(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
#             conv_bn_relu(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
#
#             conv_bn_relu(128, 256, kernel_size=3, stride=2, padding=1),
#             conv_bn_relu(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
#             conv_bn_relu(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
#
#             conv_bn_relu(256, 512, kernel_size=3, stride=2, padding=1),
#             conv_bn_relu(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
#             conv_bn_relu(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
#
#             conv_bn_relu(512, 512, kernel_size=3, stride=2, padding=1),
#             conv_bn_relu(512, 512, kernel_size=3, stride=1, padding=0),
#             conv_bn_relu(512, 512, kernel_size=1, stride=1, padding=0),
#         )
#
#         p1 = 256
#         p2 = (1, 2)
#         if mode == "hubert":
#             p1 = 32
#             p2 = (2, 2)
#
#         # Audio Encoder
#         self.audio_encoder = nn.Sequential(
#             conv_bn_relu(p1, 256, kernel_size=3, stride=1, padding=1),
#             conv_bn_relu(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
#             conv_bn_relu(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
#
#             conv_bn_relu(256, 256, kernel_size=3, stride=p2, padding=1),
#             conv_bn_relu(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
#             conv_bn_relu(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
#
#             conv_bn_relu(256, 256, kernel_size=3, stride=2, padding=2),
#             conv_bn_relu(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
#             conv_bn_relu(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
#
#             conv_bn_relu(256, 512, kernel_size=3, stride=2, padding=1),
#             conv_bn_relu(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
#             conv_bn_relu(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
#
#             conv_bn_relu(512, 512, kernel_size=3, stride=1, padding=0),
#             conv_bn_relu(512, 512, kernel_size=1, stride=1, padding=0),
#         )
#
#     def forward(self, face_sequences, audio_sequences):  # audio_sequences := (B, dim, T)
#         face_embedding = self.face_encoder(face_sequences)
#         audio_embedding = self.audio_encoder(audio_sequences)
#
#         # Flatten embeddings
#         audio_embedding = audio_embedding.view(audio_embedding.size(0), -1)
#         face_embedding = face_embedding.view(face_embedding.size(0), -1)
#
#         # Normalize embeddings
#         audio_embedding = F.normalize(audio_embedding, p=2, dim=1)
#         face_embedding = F.normalize(face_embedding, p=2, dim=1)
#
#         return audio_embedding, face_embedding


logloss = nn.BCELoss()


def cosine_loss(a, v, y):
    d = nn.functional.cosine_similarity(a, v)
    loss = logloss(d.unsqueeze(1), y)

    return loss


def train_sync_net(save_dir, dataset_dir, mode, batch_size=16, num_workers=4, lr=0.001, epoch=40, base_last_model=None):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    train_dataset = Dataset(dataset_dir, mode=mode)
    train_data_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers)
    model = SyncNet_color(mode).cuda()

    if base_last_model != None and os.path.exists(base_last_model):
        model.load_state_dict(torch.load(base_last_model))

    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=lr)
    batch_total = len(train_data_loader)
    best_lost = float('inf')
    stop_time = 0
    for e in range(epoch):
        start = time.time()
        torch.cuda.empty_cache()
        current_loss = 0
        for idx, batch in enumerate(train_data_loader):
            imgT, audioT, y, t = batch
            imgT = imgT.cuda()
            audioT = audioT.cuda()
            y = y.cuda()
            audio_embedding, face_embedding = model(imgT, audioT)
            loss = cosine_loss(audio_embedding, face_embedding, y)
            loss.backward()
            optimizer.step()
            current_loss += loss.item()
            if idx == len(train_data_loader) - 1:
                if current_loss <= best_lost:
                    stop_time = 0
                    best_lost = current_loss
                    best_model = model.state_dict()
                    torch.save(best_model, os.path.join(
                        save_dir, 'best_syncnet_model.pth'))
                else:
                    stop_time += 1
                if stop_time >= 3:
                    # 停摆时间超过3就直接交换
                    best_model = model.state_dict()
                    torch.save(best_model, os.path.join(
                        save_dir, 'best_syncnet_model.pth'))
                    best_lost = current_loss
                    stop_time = 0
                torch.save(model.state_dict(), os.path.join(
                    save_dir, 'checkpoint_syncnet_model.pth'))
            yield idx, batch_total, e, epoch, current_loss, best_lost, time.time()-start


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--dataset_dir', type=str)
    parser.add_argument('--asr', type=str)
    opt = parser.parse_args()

    # syncnet = SyncNet_color(mode=opt.asr)
    # img = torch.zeros([1,3,160,160])
    # # audio = torch.zeros([1,128,16,32])
    # audio = torch.zeros([1,16,32,32])
    # audio_embedding, face_embedding = syncnet(img, audio)
    # print(audio_embedding.shape, face_embedding.shape)
    train_sync_net(opt.save_dir, opt.dataset_dir, opt.asr)
