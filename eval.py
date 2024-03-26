import os
import copy
import time
import argparse
import random 
from tqdm import tqdm

import numpy as np

import torch

from nets import *
from dataset import *
from utils import *

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

# save path
parser.add_argument('--load_path', default='trained_models', type=str, metavar='PATH', help='path to save checkpoint (default: none)')

# specify whether seed
parser.add_argument('--seed', default=0, type=int)

# CLIP
parser.add_argument('--model_name', default='openai/clip-vit-large-patch14-336', type=str)

# client residual
parser.add_argument('--alpha', default=0.3, type=float)

# uncertainty
parser.add_argument('--uncertainty', default=0.1, type=float, help='specify the fraction of confidence for filtering ')
parser.add_argument('--threshold', default=None, type=float, help='will be computed automatically, based on args.uncertainty')


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
        _, classes_per_client, test_dataset = get_dataset(args=args, process=process)
        args.test_attribute = test_dataset.all_classes
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, 
                                                    shuffle=True, num_workers=args.num_workers, 
                                                    pin_memory=True, collate_fn=collate_fn)

        load_dir = os.path.join(args.load_path, args.dataset, 
                                'num_users_' + str(args.num_users), 
                                'num_samples_' + str(args.num_samples), 
                                'alpha_' + str(args.alpha),
                                'seed_' + str(args.seed))

    print("we are loading model from: ", load_dir)

    # get predictive threshold
    # see utils/core.py
    get_threshold(args)

    global_model.initdgatal(classes_per_client)
    
    # copy weights
    load_name = os.path.join(load_dir, 'model.checkpoint')
    if os.path.isfile(load_name):
        print("=> loading checkpoint '{}'".format(load_name))
        checkpoint = torch.load(load_name)
        
        # load adpated weights
        adapted_state_dict = checkpoint['state_dict']
        own_state = global_model.state_dict()
        for name, param in adapted_state_dict.items():
            own_state[name] = copy.deepcopy(param.data)
        global_model.load_state_dict(own_state)
        global_model.eval()
        global_model.cuda()
        global_model.float()
        # adapted_state_dict['vision_adapter.prompt_proj.bias']
        print("=> loaded checkpoint !! ")

    else:
        exit("No model found!")
        
    # global validation
    global_test(args, global_model, test_loader, tokenizer)

def global_test(args, global_model, test_loader, tokenizer):
    global_model.eval()
    if args.dataset in ['food101', 'caltech101', 'fgvc', 'flower102', 'ucf101', 'cars']:
        candidate_prompts = get_prompts(args.test_attribute)

    candidate_token_prompts = tokenizer(candidate_prompts, padding=True, return_tensors="pt")
    candidate_token_prompts = {k: v.cuda() for k, v in candidate_token_prompts.items()}
    
    text_features = global_model.model.get_text_features(**candidate_token_prompts)
    text_features = text_features / text_features.norm(dim=1, keepdim=True)

    preds_his = []
    targets_his = []

    # initialize the support set to save [textual prototypes, visual prototypes, and count of visual prototypes]
    support_set = {}
    if args.dataset in ['food101', 'caltech101', 'fgvc', 'flower102', 'ucf101', 'cars']:
        for i in range(len(args.test_attribute)):
            key = args.test_attribute[i]
            support_set[key] = [text_features[i].detach().cpu(), torch.zeros(768), 0]

    num_samples = 0
    start = time.time()
    with torch.no_grad():
        for _, batch in enumerate(tqdm(test_loader)):
            raw_images, targets = batch[0], batch[1]

            # Inference
            inputs = global_model.processor(images=raw_images, return_tensors="pt")
            inputs = {k: v.half().cuda() for k, v in inputs.items()}
            image_features = global_model.model.get_image_features(**inputs)
            
            # adapt image features
            image_features_att = global_model.img_adap(image_features)
            image_features = torch.mul(image_features_att, image_features)
        
            # normalized features
            image_features = image_features / image_features.norm(dim=1, keepdim=True)
            
            # get multi-modal prototypes
            # see utils/core.py 
            prototypes = get_prototypes(support_set)

            # get logits and perform multi-modal prototyping
            logit_scale = global_model.model.logit_scale.exp()
            logits_per_image = logit_scale * image_features @ prototypes.t()
            predictions = torch.argmax(logits_per_image, dim=-1)

            # compute entropy and update support set
            support_set = update_support_set(args, support_set, logits_per_image, args.threshold, 
                                            predictions, args.test_attribute, image_features)
            
            preds_his += predictions.tolist()
            targets_his += targets

            num_samples += len(targets)
        
        print("time per image: ", (time.time()-start)/num_samples)
        post_processing(args, preds_his, targets_his)



if __name__ == "__main__":
    main()