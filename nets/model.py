# code adapted from https://github.com/microsoft/PersonalizedFL/blob/main/fedclip/nets/models.py

import torch.nn as nn
import torch
from transformers import CLIPProcessor, CLIPModel

def get_image_features(image, model, cpreprocess, device='cuda', need_preprocess=False):
    if need_preprocess:
        image = cpreprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
    return image_features


def freeze_param(model):
    for name, param in model.named_parameters():
        param.requires_grad = False

class ClipModelat(nn.Module):
    def __init__(self, model_name=None, device='cuda', freezepy=True):
        super().__init__()

        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)

        self.model.eval()
        self.model.to(device)

        self.freezepy = freezepy
        self.device = device

    def initdgatal(self, classes_per_client):
        if self.freezepy:
            freeze_param(self.model)

        self.img_adap = nn.Sequential(
                                nn.Linear(768, 768), 
                                nn.Tanh(), 
                                nn.Linear(768, 768), 
                                nn.Softmax(dim=1)).to(self.device)

        self.task_res = nn.Parameter(torch.zeros(classes_per_client, 768))

        # initialization
        nn.init.uniform_(self.task_res.data, -5, 5)
        