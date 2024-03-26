import os
import copy
import time
import pickle
import argparse
import numpy as np
from tqdm import tqdm
import random 
import json
from tqdm import tqdm

from sklearn.metrics import precision_recall_fscore_support

import torch
import pdb
from nets import *
from dataset import *
from utils import *

from client_utils import LocalUpdate

parser = argparse.ArgumentParser()

# federated arguments (Notation for the arguments followed from paper)
parser.add_argument('--data_dir', type=str, default='YOUR PATH')
parser.add_argument('--dataset', type=str, default='caltech101', help="name of dataset")
parser.add_argument('--epochs', type=int, default=2, help="number of rounds of training")
parser.add_argument('--local_ep', type=int, default=2, help="the number of local epochs: E")
parser.add_argument('--batch_size', type=int, default=18, help="local batch size: B")
parser.add_argument('--num_workers', type=int, default=2)

# federated datasets
parser.add_argument('--num_samples', type=int, default=10, help="number of training samplpes per user")
parser.add_argument('--num_users', type=int, default=None, help="number of users: K")

# optimizer
parser.add_argument('--beta1', type=float, default=0.9)
parser.add_argument('--beta2', type=float, default=0.98)
parser.add_argument('--eps', type=float, default=1e-6)
parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')
parser.add_argument('--momentum', type=float, default=0.5, help='SGD momentum (default: 0.5)')
parser.add_argument('--weight_decay', type=float, default=0.2, help='SGD weight decay (default: 1d-4)')

# save path
parser.add_argument('--save_path', default='trained_models', type=str, metavar='PATH', help='path to save checkpoint (default: none)')

# specify whether seed
parser.add_argument('--seed', default=0, type=int)

# CLIP
parser.add_argument('--model_name', default='openai/clip-vit-large-patch14-336', type=str)

# client residual
parser.add_argument('--alpha', default=0.3, type=float)

def average_weights(w, task_similarity):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])

    for key in w_avg.keys():
        w_avg[key] = task_similarity[0] * w_avg[key]

    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += task_similarity[i] * w[i][key]
        w_avg[key] = torch.div(w_avg[key], sum(task_similarity))

    return w_avg

def get_global_text_features(args, global_model, tokenizer):
    if args.dataset in ['food101', 'caltech101', 'fgvc', 'flower102', 'ucf101', 'cars']:
        candidate_prompts = get_prompts(args.test_attribute)

    candidate_token_prompts = tokenizer(candidate_prompts, padding=True, return_tensors="pt")
    candidate_token_prompts = {k: v.cuda() for k, v in candidate_token_prompts.items()}
    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            text_features = global_model.model.get_text_features(**candidate_token_prompts).float()
            text_features = text_features / text_features.norm(dim=1, keepdim=True)

    return text_features

def get_task_similarity(global_text_features, local_text_features):
    similarity = global_text_features @ local_text_features.t() 
    similarity = similarity / torch.sqrt(torch.sum(global_text_features**2, dim=-1).unsqueeze(0) @ torch.sum(global_text_features**2, dim=-1).unsqueeze(0).t())
    # take mean over the entire matrix
    similarity = torch.mean(similarity.sum(dim=-1)).item()
    return similarity

def main():
    global args
    args = parser.parse_args()
    
    torch.backends.cudnn.deterministic = True
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.dataset in ['food101', 'caltech101', 'fgvc', 'flower102', 'ucf101', 'cars']:
        # get model
        global_model = ClipModelat(args.model_name, freezepy=True)
        process = global_model.processor
        tokenizer = global_model.processor.tokenizer

        # get data
        user_groups, classes_per_client, test_dataset = get_dataset(args=args, process=process)
        args.test_attribute = test_dataset.all_classes

        save_dir = os.path.join(args.save_path, args.dataset, 
                                'num_users_' + str(args.num_users), 
                                'num_samples_' + str(args.num_samples), 
                                'alpha_' + str(args.alpha),
                                'seed_' + str(args.seed))
    else:
        exit("dataset not implemented.")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print("we are saving model to: ", save_dir)

    global_model.initdgatal(classes_per_client)
    global_model.cuda()
    global_model.float()

    # copy weights
    global_weights = global_model.state_dict()
    global_text_features = get_global_text_features(args, global_model, tokenizer)

    # Training
    for epoch in range(args.epochs):
        local_weights, task_similarities = [], []
        print(f'\n | Global Training Round : {epoch+1} |\n')

        global_model.train()
        for idx in range(args.num_users):
            local_model = LocalUpdate(args=args, user_data=user_groups[idx], user_id=idx, process=process, tokenizer=tokenizer)
            adapter_w, text_features = local_model.update_weights(model=copy.deepcopy(global_model), global_round=epoch)

            local_weights.append(copy.deepcopy(adapter_w))
            task_similarity = get_task_similarity(global_text_features, text_features)
            task_similarities.append(task_similarity)

        # update global weights
        global_adapter_weights = average_weights(local_weights, task_similarities)

        # update global weights
        for name, param in global_weights.items():
            if name in global_adapter_weights.keys():
                global_weights[name] = global_adapter_weights[name]

        global_model.load_state_dict(global_weights)

    global_model_save_name = os.path.join(save_dir, 'model.checkpoint')
    adapter_state_dict = {}
    for name, param in global_model.state_dict().items():
        if 'img_adap' in name or 'task_res' in name:
            adapter_state_dict[name] = param

    global_model_checkpoint = {'state_dict': adapter_state_dict}
    torch.save(global_model_checkpoint, global_model_save_name)


if __name__ == "__main__":
    main()