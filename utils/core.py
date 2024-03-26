import torch
import numpy as np

def get_prototypes(support_set):
    prototypes_text = []
    prototypes_image = []
    prototypes_image_count = []
    for key in support_set:
        prototypes_text.append(support_set[key][0]) 
        prototypes_image.append(support_set[key][1])
        prototypes_image_count.append(support_set[key][2])
    
    if 0 in prototypes_image_count:
        # there are classes with only text prototype
        # should only text prototype for classification
        prototypes = torch.stack(prototypes_text).cuda()
        
    else:
        prototypes = torch.stack(prototypes_text).cuda() + torch.stack(prototypes_image).cuda()

    return prototypes

def get_threshold(args):
    if args.dataset in ['food101', 'caltech101', 'fgvc', 'flower102', 'ucf101', 'cars']:
        probs = 1 / len(args.test_attribute)
        probs_per_image = torch.tensor([probs for i in range(len(args.test_attribute))])
        max_entropy = -torch.sum(probs_per_image * torch.log(probs_per_image))
        args.threshold = max_entropy.item() * args.uncertainty

def update_support_set(args, support_set, logits_per_image, threshold, 
                        predictions, test_attribute, image_features):
    
    probs_per_image = torch.softmax(logits_per_image, dim=-1)
    entropy = -torch.sum(probs_per_image * torch.log(probs_per_image), dim=-1)
    selected_ids = torch.where((entropy < threshold) != 0)[0]
    for i in range(len(selected_ids)):
        key = predictions[selected_ids[i]].item()
        if args.dataset in ['food101', 'caltech101', 'fgvc', 'flower102', 'ucf101', 'cars']:
            key = test_attribute[key]
        prototype = support_set[key][1]
        prototype_num = support_set[key][2]
        prototype_updated = (prototype * prototype_num + image_features[selected_ids[i]].detach().cpu()) / (prototype_num + 1)

        support_set[key][1] = prototype_updated
        support_set[key][2] = prototype_num + 1

    return support_set