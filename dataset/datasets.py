from PIL import Image
import re
import torch
from torch.utils.data import Dataset
from .attributes_dict import *

class myCeleba(Dataset):
    def __init__(self, data_path=None, target_rows=None, process=None):
        self.data_path = data_path
        self.target_rows = target_rows
        self.process = process
                    
    def __len__(self):
        return len(self.data_path)
        
    def __getitem__(self, idx):
        image_set = Image.open(self.data_path[idx])
        targets = torch.tensor(self.target_rows[idx]).long()

        return image_set, targets

class myCaltech(Dataset):
    def __init__(self, data_path=None, all_classes=None, process=None):
        self.data_path = data_path
        self.process = process
        self.all_classes = all_classes # all classes each client might have
                    
    def __len__(self):
        return len(self.data_path)
        
    def __getitem__(self, idx):
        image_set = Image.open(self.data_path[idx])
        targets = self.data_path[idx].split('/')[-2]

        return image_set, targets

class myUCF(Dataset):
    def __init__(self, data_path=None, all_classes=None, process=None):
        self.data_path = data_path
        self.process = process
        self.all_classes = all_classes # all classes each client might have
                    
    def __len__(self):
        return len(self.data_path)
        
    def __getitem__(self, idx):
        image_set = Image.open(self.data_path[idx])
        regex = r"[A-Z][a-z]*[^A-Z]"
        
        targets = self.data_path[idx].split('/')[-2]
        match = re.findall(regex, targets) 
        targets = ' '.join(match)

        return image_set, targets
    
class myFGVC(Dataset):
    def __init__(self, data_path=None, all_classes=None, label_dict=None, process=None):
        self.data_path = data_path
        self.process = process
        self.label_dict = label_dict
        self.all_classes = all_classes # all classes each client might have

    def __len__(self):
        return len(self.data_path)
        
    def __getitem__(self, idx):
        image_set = Image.open(self.data_path[idx])
        image_id = self.data_path[idx].split('/')[-1].split('.')[0]
        targets = self.label_dict[image_id]
        
        return image_set, targets

class myFlower(Dataset):
    def __init__(self, data_path=None, all_classes=None, process=None):
        self.data_path = data_path
        self.process = process
        self.all_classes = all_classes # all classes each client might have

    def __len__(self):
        return len(self.data_path)
        
    def __getitem__(self, idx):
        image_set = Image.open(self.data_path[idx])
        class_id = self.data_path[idx].split('/')[-2]
        targets = flower_attribute_dict[class_id]
        
        return image_set, targets

class myPET(Dataset):
    def __init__(self, data_path=None, all_classes=None, process=None):
        self.data_path = data_path
        self.process = process
        self.all_classes = all_classes # all classes each client might have

    def __len__(self):
        return len(self.data_path)
        
    def __getitem__(self, idx):
        image_set = Image.open(self.data_path[idx])
        targets = ' '.join(self.data_path[idx].split('/')[-1].split('_')[:-1])
        
        return image_set, targets