import time
import numpy as np
import os
import random
import torchvision.transforms as transforms
from utils.dataset_utils import split_data, save_file
from os import path
from PIL import Image
from torch.utils.data import DataLoader, Dataset


def listdir_nohidden(path):
    """List non-hidden items in a directory.

    Args:
         path (str): directory path.
    """
    return [f for f in os.listdir(path) if not f.startswith('.')]


def read_office_caltech10_data(dataset_path, domain_name):
    data_paths = []
    data_labels = []
    domain_dir = path.join(dataset_path,"office-10", domain_name)
    class_names = listdir_nohidden(domain_dir)
    class_names.sort()
    for label, class_name in enumerate(class_names):
        class_dir = path.join(domain_dir, class_name)
        item_names = listdir_nohidden(class_dir)
        for item_name in item_names:
            item_path = path.join(class_dir, item_name)
            data_paths.append(item_path)
            data_labels.append(label)
    return data_paths, data_labels

# https://github.com/FengHZ/KD3A/blob/master/datasets/OfficeCaltech10.py
class OfficeCaltech(Dataset):
    def __init__(self, data_paths, data_labels, transforms, domain_name):
        super(OfficeCaltech, self).__init__()
        self.data_paths = data_paths
        self.data_labels = data_labels
        self.transforms = transforms
        self.domain_name = domain_name

    def __getitem__(self, index):
        img = Image.open(self.data_paths[index])
        img = img.convert("RGB")
        label = self.data_labels[index]
        img = self.transforms(img)

        return img, label

    def __len__(self):
        return len(self.data_paths)


def get_office_caltech10_dloader(dataset_path, domain_name):
    data_paths, data_labels = read_office_caltech10_data(dataset_path, domain_name)
    transforms_train = transforms.Compose([
        transforms.RandomResizedCrop(64, scale=(0.75, 1)), 
        transforms.RandomHorizontalFlip(),    
        transforms.ToTensor() 
    ])
    transforms_test = transforms.Compose([
        transforms.Resize((64, 64)),  
        transforms.ToTensor()
    ])
    train_dataset = OfficeCaltech(data_paths, data_labels, transforms_train, domain_name)
    test_dataset = OfficeCaltech(data_paths, data_labels, transforms_test, domain_name)
    train_dloader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)
    test_dloader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
    return train_dloader, test_dloader


random.seed(1)
np.random.seed(1)
data_path = "Office_10/"
dir_path = "Office_10/"

# Allocate data to users
def generate_Office_10(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


    # Setup directory for train/test data
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"   

    if not os.path.exists(train_path):
        os.makedirs(train_path)
    if not os.path.exists(test_path):
        os.makedirs(test_path)

    root = data_path+"rawdata"


    # if not os.path.exists(root):
    #     os.makedirs(root)
    #     os.system(f'wget https://drive.google.com/drive/folders/1SJLhRiXbAwNpbgFXIg2KQRBZA5ZYQ62W?usp=share_link -P {root}') 
        # os.system(f'unzip {root}/ -d {root}')  


    X,y=[],[]
    domains=['amazon','caltech','dslr','webcam']
    for d in domains:
        train_loader, test_loader = get_office_caltech10_dloader(root, d)  

        for _, tt in enumerate(train_loader):  
            train_data, train_label = tt      
        for _, tt in enumerate(test_loader):
            test_data, test_label = tt       

        dataset_image = []
        dataset_label = []

        dataset_image.extend(train_data.cpu().detach().numpy())
        dataset_image.extend(test_data.cpu().detach().numpy())    
        dataset_label.extend(train_label.cpu().detach().numpy())
        dataset_label.extend(test_label.cpu().detach().numpy())   

        X.append(np.array(dataset_image))        
        y.append(np.array(dataset_label))        
    

    labelss = []                     
    for yy in y:
        labelss.append(len(set(yy)))  
    num_clients = len(y)               
    print(f'Number of labels: {labelss}')    
    print(f'Number of clients: {num_clients}')

    statistic = [[] for _ in range(num_clients)]
    for client in range(num_clients):
        for i in np.unique(y[client]):       
            statistic[client].append((int(i), int(sum(y[client]==i)))) 


    train_data, test_data = split_data(X, y)        
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, max(labelss), 
        statistic, None, None, None)
    

if __name__ == "__main__":
    generate_Office_10(dir_path)