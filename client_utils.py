import copy
import numpy as np 

import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import *
from utils import *

class LocalUpdate(object):
    def __init__(self, args, user_data, user_id, process, tokenizer):
        self.args = args
        self.client_id = user_id
        self.process = process
        self.client_attribute = user_data[0]
        self.train_loader, self.val_loader = self.train_val_split(user_data)
        self.tokenizer = tokenizer

    def update_weights(self, model, global_round):
        # Set mode to train model
        model.train()
        epoch_loss = []

        optimizer = self.build_optimizer(model)
        loss_img = nn.CrossEntropyLoss()
        loss_txt = nn.CrossEntropyLoss()

        scaler = torch.cuda.amp.GradScaler()
        
        best_acc = 0
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, batch in enumerate(self.train_loader):
                raw_images, targets = batch[0], batch[1]
                batch_size = len(targets)
                
                prompts, candidate_prompts = self.get_batch_prompts(targets)
                token_prompts = self.tokenizer(prompts, padding=True, return_tensors="pt")
                token_prompts = {k: v.cuda() for k, v in token_prompts.items()}

                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    inputs = model.processor(images=raw_images, return_tensors="pt")
                    inputs = {k: v.half().cuda() for k, v in inputs.items()}
                    image_features = model.model.get_image_features(**inputs)
                    text_features = model.model.get_text_features(**token_prompts)
                    
                    # adapt image features
                    image_features_att = model.img_adap(image_features)
                    image_features = torch.mul(image_features_att, image_features)
                    
                    # add client residual
                    num_targets = self.get_numerical_targets(targets)
                    text_features = text_features + self.args.alpha * model.task_res[num_targets]

                    # normalized features
                    image_features = image_features / image_features.norm(dim=1, keepdim=True)
                    text_features = text_features / text_features.norm(dim=1, keepdim=True)

                    logit_scale = model.model.logit_scale.exp()
                    logits_per_image = logit_scale * image_features @ text_features.t()
                    logits_per_text = logits_per_image.t()

                    ground_truth = torch.arange(batch_size, dtype=torch.long).cuda()

                    loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth))/2

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                batch_loss.append(loss.item())
                optimizer.zero_grad()

                candidate_token_prompts = self.tokenizer(candidate_prompts, padding=True, return_tensors="pt")
                candidate_token_prompts = {k: v.cuda() for k, v in candidate_token_prompts.items()}
                with torch.no_grad():
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        # Inference with clip
                        image_features = model.model.get_image_features(**inputs).float()
                        text_features = model.model.get_text_features(**candidate_token_prompts).float()
                        
                        # adapt image features
                        image_features_att = model.img_adap(image_features)
                        image_features_att = torch.mul(image_features_att, image_features)

                        # add client residual
                        text_features = text_features + self.args.alpha * model.task_res

                        # normalized features   
                        image_features = image_features / image_features.norm(dim=1, keepdim=True)
                        text_features = text_features / text_features.norm(dim=1, keepdim=True)

                        logit_scale = model.model.logit_scale.exp()
                        logits_per_image = logit_scale * image_features @ text_features.t()
                        predictions = torch.argmax(logits_per_image, dim=-1)

                    acc = self.post_processing(predictions.tolist(), targets, candidate_prompts)
                    print('Global Round: {}, Local Epoch: {} [{}/{}],   Loss: {:.4f},  Acc: {:.4f}'.format(
                        global_round, iter, batch_idx, len(self.train_loader), loss.item(), acc))
                    
                epoch_loss.append(sum(batch_loss)/len(batch_loss))

            acc = self.validate(model)
            adapter_state_dict = None
            if acc > best_acc:
                best_acc = acc
                adapter_state_dict = {}
                for name, param in model.state_dict().items():
                    if 'img_adap' in name:
                        adapter_state_dict[name] = copy.deepcopy(param)

        if not adapter_state_dict:
            adapter_state_dict = {}
            for name, param in model.state_dict().items():
                if 'img_adap' in name:
                    adapter_state_dict[name] = copy.deepcopy(param)
                    
        return adapter_state_dict, text_features

    def validate(self, model):
        """ Returns the validate accuracy and loss.
        """
        model.float()
        model.eval()
        
        targets_his = []
        predictions_his = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader): 
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    raw_images, targets = batch[0], batch[1]
                    batch_size = len(targets)
                    
                    _, candidate_prompts = self.get_batch_prompts(targets)
                    candidate_token_prompts = self.tokenizer(candidate_prompts, padding=True, return_tensors="pt")
                    candidate_token_prompts = {k: v.cuda() for k, v in candidate_token_prompts.items()}

                    inputs = model.processor(images=raw_images, return_tensors="pt")
                    inputs = {k: v.half().cuda() for k, v in inputs.items()}
                    image_features = model.model.get_image_features(**inputs)
                    text_features = model.model.get_text_features(**candidate_token_prompts)
                
                    # adapt image features
                    image_features_att = model.img_adap(image_features)
                    image_features = torch.mul(image_features_att, image_features)

                    # add client residual
                    text_features = text_features + self.args.alpha * model.task_res

                    # normalized features
                    image_features = image_features / image_features.norm(dim=1, keepdim=True)
                    text_features = text_features / text_features.norm(dim=1, keepdim=True)

                    logit_scale = model.model.logit_scale.exp()
                    logits_per_image = logit_scale * image_features @ text_features.t()
                    predictions = torch.argmax(logits_per_image, dim=-1)
                    
                    targets_his += targets
                    predictions_his += predictions.tolist()

        acc = self.post_processing(predictions_his, targets_his, candidate_prompts)
                    
        print("Client ID: {}, ACC {:.3f}".format(self.client_id, acc))
        print("=" * 50)

        return acc

    def build_optimizer(self, model):
        trainable_parameters = 0
        all_parameters = 0
        params = [{'params': [p for (n, p) in model.named_parameters() if 'img_adap' in n or 'task_res' in n]}]
        for name, param in model.named_parameters():
            all_parameters += param.numel()
            if 'img_adap' in name or 'task_res' in name:
                trainable_parameters += param.numel()
                param.requires_grad = True
                
            else:
                param.requires_grad = False
        
        print("num trainable: {}, fraction: {:.4f}%".format(trainable_parameters, (trainable_parameters/all_parameters)*100))

        optimizer = torch.optim.Adam(params, lr=self.args.lr, betas=(self.args.beta1, self.args.beta2), 
                                    eps=self.args.eps, weight_decay=self.args.weight_decay)

        return optimizer

    def train_val_split(self, user_data):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        if self.args.dataset in ['food101', 'caltech101']:
            num_samples = len(user_data[1])
            np.random.seed(0)
            idxs = np.random.choice(num_samples, num_samples, replace=False)

            idxs_train = idxs[:int(0.8*num_samples)]
            idxs_val = idxs[int(0.8*num_samples):]

            inputs_arr = np.array(user_data[1])
            train_dataset = myCaltech(data_path=inputs_arr[idxs_train], 
                                    process=self.process)
            
            val_dataset = myCaltech(data_path=inputs_arr[idxs_val], 
                                    process=self.process)
        

        elif self.args.dataset in ['flower102']:
            num_samples = len(user_data[1])
            np.random.seed(0)
            idxs = np.random.choice(num_samples, num_samples, replace=False)

            idxs_train = idxs[:int(0.8*num_samples)]
            idxs_val = idxs[int(0.8*num_samples):]

            inputs_arr = np.array(user_data[1])
            train_dataset = myFlower(data_path=inputs_arr[idxs_train], 
                                    process=self.process)
            
            val_dataset = myFlower(data_path=inputs_arr[idxs_val], 
                                    process=self.process)
        
        elif self.args.dataset in ['ucf101']:
            num_samples = len(user_data[1])
            np.random.seed(0)
            idxs = np.random.choice(num_samples, num_samples, replace=False)

            idxs_train = idxs[:int(0.8*num_samples)]
            idxs_val = idxs[int(0.8*num_samples):]

            inputs_arr = np.array(user_data[1])
            train_dataset = myUCF(data_path=inputs_arr[idxs_train], 
                                    process=self.process)
            
            val_dataset = myUCF(data_path=inputs_arr[idxs_val], 
                                    process=self.process)
        
        elif self.args.dataset in ['fgvc', 'cars']:
            num_samples = len(user_data[1])
            np.random.seed(0)
            idxs = np.random.choice(num_samples, num_samples, replace=False)

            idxs_train = idxs[:int(0.8*num_samples)]
            idxs_val = idxs[int(0.8*num_samples):]

            inputs_arr = np.array(user_data[1])
    
            train_dataset = myFGVC(data_path=inputs_arr[idxs_train],
                                    label_dict = user_data[2],
                                    process=self.process)

            val_dataset = myFGVC(data_path=inputs_arr[idxs_val],
                                label_dict = user_data[2],
                                process=self.process)
        
        train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=self.args.batch_size, shuffle=False, collate_fn=collate_fn)

        return train_loader, val_loader 

    def get_batch_prompts(self, targets):
        batch_prompts = []
        if self.args.dataset in ['food101', 'caltech101', 'fgvc', 'flower102', 'ucf101', 'cars']:
            candidate_prompts = get_prompts(self.client_attribute)
            for target in targets:
                batch_prompts.append("A photo of " + target + '.')

        return batch_prompts, candidate_prompts

    def get_numerical_targets(self, targets):
        num_targets = []
        if self.args.dataset in ['food101', 'caltech101', 'fgvc', 'flower102', 'ucf101', 'cars']:
            for target in targets:
                num_targets.append(self.client_attribute.index(target))

        return num_targets

    def post_processing(self, predictions, targets, candidate_prompts):
        if self.args.dataset in ['food101', 'caltech101', 'fgvc', 'flower102', 'ucf101', 'cars']:
            num_samples = len(predictions)
            correct = 0
            
            for i in range(num_samples):
                prediction = predictions[i]
                target = targets[i]
                if target in candidate_prompts[prediction]:
                    correct += 1
            
            acc = correct / num_samples

        return acc