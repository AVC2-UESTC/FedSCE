import numpy as np
import os
import sys
import random
import torch
import torchvision
import torchvision.transforms as transforms
from utils.dataset_utils import check, separate_data, split_data, save_file


random.seed(1)
np.random.seed(1)
num_clients = 20
num_classes = 10
dir_path = "Cifar10/"


# Allocate data to users
def generate_cifar10(dir_path, num_clients, num_classes, niid, balance, partition):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
    # Setup directory for train/test data
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"

    if check(config_path, train_path, test_path, num_clients, num_classes, niid, balance, partition):
        return
        
    # Get Cifar10 data
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(      
        root=dir_path+"rawdata", train=True, download=True, transform=transform)         #数据集将存储在 dir_path 变量指定的目录下的 rawdata 子目录中。trainset 是一个 torchvision.datasets.CIFAR10 类的实例。下载的数据保持其原始格式，没有应用任何 transform 操作。只有在你实际访问数据时，变换才会被应用。例如，通过 DataLoader 迭代数据时，或者直接访问 trainset[idx] 时，transform 才会对图像进行处理。
    testset = torchvision.datasets.CIFAR10(
        root=dir_path+"rawdata", train=False, download=True, transform=transform)   #情况1: trainset.data 是一个 NumPy 数组，trainset.targets 是一个列表。情况1: trainset.data 中的图像数据未经过任何预处理，仍然是原始图像数据。情况1: trainset.data 是原始的 8-bit 图像数据，trainset.targets 是整数标签。
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=len(trainset.data), shuffle=False)      #batch_size 设置为数据集的大小，表示一次性将整个训练数据集加载到内存中。
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=len(testset.data), shuffle=False)

    for _, train_data in enumerate(trainloader, 0):
        trainset.data, trainset.targets = train_data   #情况2: trainset.data 和 trainset.targets 都是 PyTorch 张量。情况2: trainset.data 中的图像数据已经经过了预处理变换（如 transforms.ToTensor()、transforms.Normalize() 等）。情况2: trainset.data 是浮点型张量数据（经过归一化等处理），trainset.targets 也是整数型张量。
    for _, test_data in enumerate(testloader, 0):
        testset.data, testset.targets = test_data

    dataset_image = []
    dataset_label = []

    dataset_image.extend(trainset.data.cpu().detach().numpy())
    dataset_image.extend(testset.data.cpu().detach().numpy())
    dataset_label.extend(trainset.targets.cpu().detach().numpy())
    dataset_label.extend(testset.targets.cpu().detach().numpy())
    dataset_image = np.array(dataset_image)   #dataset_image是np格式，包含所有的样本（train）
    dataset_label = np.array(dataset_label)

    # dataset = []
    # for i in range(num_classes):
    #     idx = dataset_label == i
    #     dataset.append(dataset_image[idx])

    X, y, statistic = separate_data((dataset_image, dataset_label), num_clients, num_classes,  
                                    niid, balance, partition, class_per_client=2)
    train_data, test_data = split_data(X, y)
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes, 
        statistic, niid, balance, partition)


if __name__ == "__main__":
    niid = True if sys.argv[1] == "noniid" else False
    balance = True if sys.argv[2] == "balance" else False
    partition = sys.argv[3] if sys.argv[3] != "-" else None

    generate_cifar10(dir_path, num_clients, num_classes, niid, balance, partition)