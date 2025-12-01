import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, use_res_connect, expand_ratio=6):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.use_res_connect = use_res_connect

        self.conv = nn.Sequential(
            nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            # nn.ReLU(inplace=False),
            nn.LeakyReLU(inplace=False),
            nn.Conv2d(inp * expand_ratio,
                      inp * expand_ratio,
                      3,
                      stride,
                      1,
                      groups=inp * expand_ratio,
                      bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            # nn.ReLU(inplace=False),
            nn.LeakyReLU(inplace=False),
            nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
            nn.LeakyReLU(inplace=False),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class DoubleConvDW(nn.Module):

    def __init__(self, in_channels, out_channels, stride=2):
        super(DoubleConvDW, self).__init__()
        self.double_conv = nn.Sequential(
            InvertedResidual(in_channels, out_channels, stride=stride, use_res_connect=False, expand_ratio=2),
            InvertedResidual(out_channels, out_channels, stride=1, use_res_connect=True, expand_ratio=2)
        )

    def forward(self, x):
        return self.double_conv(x)


class InConvDw(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InConvDw, self).__init__()
        self.inconv = nn.Sequential(
            InvertedResidual(in_channels, out_channels, stride=1, use_res_connect=False, expand_ratio=2)
        )

    def forward(self, x):
        return self.inconv(x)


class Down(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            DoubleConvDW(in_channels, out_channels, stride=2)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConvDW(in_channels, out_channels, stride=1)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.shape[2] - x1.shape[2]
        diffX = x2.shape[3] - x1.shape[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x1, x2], axis=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class AudioConvWenet(nn.Module):
    def __init__(self):
        super(AudioConvWenet, self).__init__()
        # ch = [16, 32, 64, 128, 256]   # if you want to run this model on a mobile device, use this. 
        ch = [32, 64, 128, 256, 512]
        self.conv1 = InvertedResidual(ch[3], ch[3], stride=1, use_res_connect=True, expand_ratio=2)
        self.conv2 = InvertedResidual(ch[3], ch[3], stride=1, use_res_connect=True, expand_ratio=2)

        self.conv3 = nn.Conv2d(ch[3], ch[3], kernel_size=3, padding=1, stride=(1, 2))
        self.bn3 = nn.BatchNorm2d(ch[3])

        self.conv4 = InvertedResidual(ch[3], ch[3], stride=1, use_res_connect=True, expand_ratio=2)

        self.conv5 = nn.Conv2d(ch[3], ch[4], kernel_size=3, padding=3, stride=2)
        self.bn5 = nn.BatchNorm2d(ch[4])
        self.relu = nn.ReLU()

        self.conv6 = InvertedResidual(ch[4], ch[4], stride=1, use_res_connect=True, expand_ratio=2)
        self.conv7 = InvertedResidual(ch[4], ch[4], stride=1, use_res_connect=True, expand_ratio=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = self.relu(self.bn5(self.conv5(x)))
        x = self.conv6(x)
        x = self.conv7(x)
        return x


class AudioConvHubert(nn.Module):
    def __init__(self):
        super(AudioConvHubert, self).__init__()
        # ch = [40, 80, 160, 320, 640]
        # ch = [16*3, 32*3, 64*3, 128*3, 256*3] #
        ch = [32, 64, 128, 256, 512]  # 中
        # ch = [16, 32, 64, 128, 256]
        # ch = [24, 48, 96, 192, 384]
        # ch = [32*2, 64*2, 128*2, 256*2, 512*2]
        # ch = [32*4, 64*4, 128*4, 256*4, 512*4]
        self.conv1 = InvertedResidual(32, ch[1], stride=1, use_res_connect=False, expand_ratio=2)
        self.conv2 = InvertedResidual(ch[1], ch[2], stride=1, use_res_connect=False, expand_ratio=2)
        self.conv3 = nn.Conv2d(ch[2], ch[3], kernel_size=3, padding=1, stride=(2, 2))
        self.bn3 = nn.BatchNorm2d(ch[3])
        self.conv4 = InvertedResidual(ch[3], ch[3], stride=1, use_res_connect=True, expand_ratio=2)
        self.conv5 = nn.Conv2d(ch[3], ch[4], kernel_size=3, padding=3, stride=2)
        self.bn5 = nn.BatchNorm2d(ch[4])
        self.relu = nn.LeakyReLU()
        self.conv6 = InvertedResidual(ch[4], ch[4], stride=1, use_res_connect=True, expand_ratio=2)
        self.conv7 = InvertedResidual(ch[4], ch[4], stride=1, use_res_connect=True, expand_ratio=2)
        self.bn7 = nn.BatchNorm2d(ch[4])
        self.relu7 = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = self.relu(self.bn5(self.conv5(x)))
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.relu7(self.bn7(x))
        return x


# 注意力机制
class CrossAttention(nn.Module):
    def __init__(self, in_channels):
        super(CrossAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x, y):
        batch_size, channels, height, width = x.size()
        proj_query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(y).view(batch_size, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)
        proj_value = self.value_conv(y).view(batch_size, -1, width * height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)
        out = self.gamma * out + x
        return out


# MLP融合
import torch
import torch.nn as nn

class MLPFusion(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(MLPFusion, self).__init__()
        self.fc1 = nn.Linear(in_channels * 2, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)  # 添加BatchNorm1d层
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)  # 添加BatchNorm1d层
        self.relu = nn.LeakyReLU()

    def forward(self, x, y):
        batch_size, channels, height, width = x.size()
        # 将x和y展平并转换成(B, H*W, C)的形式
        x = x.view(batch_size, channels, -1).permute(0, 2, 1)
        y = y.view(batch_size, channels, -1).permute(0, 2, 1)
        # 拼接x和y，并通过全连接层处理
        fused = torch.cat([x, y], dim=-1)  # (B, H*W, 2*C)
        fused = self.fc1(fused)  # (B, H*W, hidden_channels)
        fused = self.bn1(fused.permute(0, 2, 1)).permute(0, 2, 1)  # 应用BatchNorm1d，并调整维度以适应输入要求
        fused = self.relu(fused)  # (B, H*W, hidden_channels)
        fused = self.fc2(fused)  # (B, H*W, out_channels)
        fused = self.bn2(fused.permute(0, 2, 1)).permute(0, 2, 1)  # 同样地，应用BatchNorm1d并调整维度
        # 转换回原始形状 (B, out_channels, H, W)，这里确保out_channels等于channels * 2
        fused = fused.permute(0, 2, 1).view(batch_size, -1, height, width)
        return fused


import torch.nn as nn
class AttentionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttentionBlock, self).__init__()
        self.cross_attention = CrossAttention(in_channels)
        self.attention_adjust_p_1 = nn.Conv2d(out_channels, in_channels, kernel_size=1)
        self.attention_adjust_b_1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.lru = nn.LeakyReLU()
    def forward(self, x, audio_feat, tx):
        ox = self.attention_adjust_p_1(x)
        ox = self.cross_attention(ox, audio_feat)
        ox = self.attention_adjust_b_1(ox)
        ox = ox + tx
        ox = self.bn(ox)
        ox = self.lru(ox)
        return ox


class Model(nn.Module):
    def __init__(self, n_channels=6, mode='hubert', n_blocks=4):
        super(Model, self).__init__()
        self.n_channels = n_channels
        ch = [32, 64, 128, 256, 512]
        # ch = [16, 32, 64, 128, 256]
        # ch = [24, 48, 96, 192, 384]
        if mode == 'hubert':
            self.audio_model = AudioConvHubert()
        elif mode == 'wenet':
            self.audio_model = AudioConvWenet()

        self.fuse_conv = nn.Sequential(
            DoubleConvDW(ch[4] * 2, ch[4], stride=1),
            DoubleConvDW(ch[4], ch[3], stride=1)
        )
        self.inc = InConvDw(n_channels, ch[0])
        self.down1 = Down(ch[0], ch[1])
        self.down2 = Down(ch[1], ch[2])
        self.down3 = Down(ch[2], ch[3])
        self.down4 = Down(ch[3], ch[4])

        self.up1 = Up(ch[4], ch[3] // 2)
        self.up2 = Up(ch[3], ch[2] // 2)
        self.up3 = Up(ch[2], ch[1] // 2)
        self.up4 = Up(ch[1], ch[0])
        self.outc = OutConv(ch[0], 3)
        self.outc_bn = nn.BatchNorm2d(3)

        self.mlp_fusion = MLPFusion(ch[4], ch[4] * 2, ch[4] * 2)

        # 创建多个注意力模块
        self.attention_blocks = nn.ModuleList([
            AttentionBlock(ch[4], ch[4] * 2) for _ in range(n_blocks)
        ])

        self.bn_kx = nn.BatchNorm2d(ch[4]*2)
        self.bn_tx = nn.BatchNorm2d(ch[4]*2)
        self.lru_kx = nn.LeakyReLU()

    def forward(self, x, audio_feat):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)  # x5,归一 rule 之后
        # 处理音频特征向量并调整其通道数以匹配x5
        audio_feat = self.audio_model(audio_feat)
        # 使用mlp融合视觉特征和音频特征
        tx = torch.cat([x5, audio_feat], dim=1)
        fused_feat = self.mlp_fusion(x5, audio_feat)
        tx = tx + fused_feat
        tx = self.bn_tx(tx)
        # 应用注意力模块
        ox = tx
        kx = tx
        # print(ox.shape,audio_feat.shape,tx.shape)
        for block in self.attention_blocks:
            ox = block(ox, audio_feat, tx)
            kx = ox + kx
        # 直接使用融合后的特征
        kx = self.bn_kx(kx)
        kx = self.lru_kx(kx)
        x5 = self.fuse_conv(kx)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        out = self.outc(x)
        out = self.outc_bn(out)
        out = F.sigmoid(out)
        return out



if __name__ == '__main__':
    import torch
    import time
    import numpy as np
    
    print("开始模型测试...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 创建模型实例
    net = Model(6, mode='hubert').eval().to(device)
    
    # 加载预训练权重
    try:
        checkpoint = torch.load(r"I:\BaiduNetdiskDownload\数字人智播系统\human-infer2\modules\dh\000\checkpoint\30_0.0150.pth", map_location=device)
        if 'state_dict' in checkpoint:
            net.load_state_dict(checkpoint['state_dict'])
        else:
            net.load_state_dict(checkpoint)
        print("成功加载预训练权重")
    except Exception as e:
        print(f"加载权重时出错: {str(e)}")
    
    print("模型已加载")

    # 创建测试数据
    batch_size = 8
    test_batches = 1000  # 每轮测试的批次数
    test_rounds = 3      # 测试轮数
    
    img = torch.randn(batch_size, 6, 160, 160).to(device)
    audio = torch.randn(batch_size, 32, 32, 32).to(device)
    print(f"测试数据形状:")
    print(f"图像输入: {img.shape}")
    print(f"音频输入: {audio.shape}")

    # 预热GPU
    print("\n预热GPU...")
    with torch.no_grad():
        for _ in range(50):
            _ = net(img, audio)
    torch.cuda.synchronize()
    
    # 开始速度测试
    print("\n开始速度测试...")
    total_frames = 0
    start_time = time.time()
    
    for round in range(test_rounds):
        print(f"\n开始第{round + 1}轮测试...")
        round_start = time.time()
        
        with torch.no_grad():
            for batch in range(test_batches):
                output = net(img, audio)
                
                if (batch + 1) % 100 == 0:
                    torch.cuda.synchronize()
                    current_time = time.time()
                    elapsed_time = current_time - start_time
                    frames = (batch + 1 + round * test_batches) * batch_size
                    fps = frames / elapsed_time
                    print(f"当前进度: {batch + 1}/{test_batches}, 速度: {fps:.2f} FPS")
        
        torch.cuda.synchronize()
        round_time = time.time() - round_start
        round_frames = test_batches * batch_size
        round_fps = round_frames / round_time
        total_frames += round_frames
        print(f"第{round + 1}轮完成 - 速度: {round_fps:.2f} FPS")

    # 计算总体性能
    total_time = time.time() - start_time
    average_fps = total_frames / total_time
    
    print(f"\n测试完成:")
    print(f"总批次数: {test_rounds * test_batches}")
    print(f"总帧数: {total_frames}")
    print(f"总耗时: {total_time:.2f}秒")
    print(f"平均速度: {average_fps:.2f} FPS")

