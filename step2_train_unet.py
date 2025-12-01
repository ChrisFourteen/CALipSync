import os
import gc
import time
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import torchvision.models as models
from dataset.dataset import MyDataset
from module.unet import Model

class PerceptualLoss():
    def contentFunc(self, vgg_path):
        conv_3_3_layer = 14
        cnn = models.vgg19()
        cnn.load_state_dict(torch.load(vgg_path))
        cnn = cnn.features
        cnn = cnn.cuda()
        model = nn.Sequential()
        model = model.cuda()
        for i, layer in enumerate(list(cnn)):
            model.add_module(str(i), layer)
            if i == conv_3_3_layer:
                break
        return model

    def __init__(self, loss, vgg_path):
        self.criterion = loss
        self.contentFunc = self.contentFunc(vgg_path)

    def get_loss(self, fakeIm, realIm):
        f_fake = self.contentFunc.forward(fakeIm)
        f_real = self.contentFunc.forward(realIm)
        f_real_no_grad = f_real.detach()
        loss = self.criterion(f_fake, f_real_no_grad)
        return loss

def train_digital_model(
    dataset_dir: str,
    save_dir: str, 
    vgg_path: str,
    unet_checkpoint: str = None,
    asr: str = "hubert",
    batch_size: int = 16,
    num_workers: int = 4,
    lr: float = 0.001,
    epochs: int = 20
):
    """
    训练数字模型的主函数
    
    Args:
        dataset_dir: 数据集目录
        save_dir: 模型保存目录
        vgg_path: VGG19预训练权重路径
        unet_checkpoint: UNet预训练模型路径（可选）
        asr: ASR模型类型，默认为"hubert"
        batch_size: 批次大小，默认为16
        num_workers: 数据加载线程数，默认为4
        lr: 学习率，默认为0.001
        epochs: 训练轮数，默认为20
    """
    print("初始化训练环境...")
    torch.cuda.empty_cache()
    
    # 创建保存目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"创建保存目录: {save_dir}")
    
    # 初始化模型
    net = Model(6, asr).cuda()
    # 加载预训练模型（如果提供）
    if unet_checkpoint and os.path.exists(unet_checkpoint):
        print(f"加载预训练模型: {unet_checkpoint}")
        net.load_state_dict(torch.load(unet_checkpoint))
    
    # 初始化训练组件
    content_loss = PerceptualLoss(torch.nn.MSELoss(), vgg_path)
    dataset = MyDataset(dataset_dir, asr)
    train_dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers
    )
    
    optimizer = optim.Adam(net.parameters(), lr=lr)
    criterion = nn.L1Loss()
    
    print("开始训练...")
    
    for epoch in range(epochs):
        epoch_start = time.time()
        net.train()
        train_total = len(train_dataloader)
        epoch_loss = 0.0
        
        for idx, batch in enumerate(train_dataloader):
            imgs, labels, audio_feat = batch
            imgs = imgs.cuda()
            labels = labels.cuda()
            audio_feat = audio_feat.cuda()
            
            # 前向传播
            preds = net(imgs, audio_feat)
            
            # 计算损失
            loss_perceptual = content_loss.get_loss(preds, labels)
            loss_pixel = criterion(preds, labels)
            loss = loss_pixel + loss_perceptual * 0.1
            
            # 反向传播
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # 打印进度
            if idx % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Batch {idx}/{train_total} | Loss {loss.item():.6f}")
        
        # 只在最后一轮保存模型
        if epoch == epochs - 1:
            checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save(net.state_dict(), checkpoint_path)
            print(f"模型已保存到: {checkpoint_path}")
        
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch+1} 完成 | 用时 {epoch_time:.2f}秒 | 平均损失 {epoch_loss/len(train_dataloader):.6f}\n")
    
    print("训练完成!")
    
    # 清理显存
    torch.cuda.empty_cache()
    gc.collect()