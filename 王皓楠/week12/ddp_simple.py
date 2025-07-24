
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

# 设置分布式环境
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

# 清理分布式环境
def cleanup():
    dist.destroy_process_group()

# 定义训练循环
def train(rank, world_size):
    setup(rank, world_size)

    # 定义模型并将其移动到对应的 GPU 设备端
    model = models.resnet50().to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    # 损失函数及优化器
    criterion = nn.CrossEntropyLoss().to(rank)
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.01)

    # 定义数据集Dataset的转换和图像增强
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    # 分布式训练采样器
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=32)

    # 在训练开始时创建一次
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(10):
        ddp_model.train()
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(rank), labels.to(rank)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = ddp_model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), 1)

            scaler.step(optimizer)
            scaler.update()

#             loss.backward()
#             optimizer.step()
            print(f"Rank {rank}, Epoch {epoch}, Loss: {loss.item()}")

    cleanup()

def main():
    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()
