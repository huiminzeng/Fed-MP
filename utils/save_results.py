import json
import os
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

def post_processing(args, predictions, targets):
    if args.dataset in ['food101', 'caltech101', 'fgvc', 'flower102', 'ucf101', 'cars']:
        num_samples = len(predictions)
        preds_his = np.array(predictions)
        targets_his = []
        for i in range(num_samples):
            target = targets[i]
            targets_his.append(args.test_attribute.index(target))
        targets_his = np.array(targets_his)

    precision, recall, f1, _ = precision_recall_fscore_support(targets_his, preds_his, average='macro')
    accuracy = np.mean(preds_his == targets_his)

    print("Global Test Accuracy: {:.4f}, Precision: {:.4f}, Reall: {:.4f}, F1: {:.4f}".format(
        accuracy, precision, recall, f1))
    
    if args.dataset in ['food101', 'caltech101', 'fgvc', 'flower102', 'ucf101', 'cars']:
        save_dir = os.path.join('experiments', args.dataset, 
                                'num_users_' + str(args.num_users), 
                                'num_samples_' + str(args.num_samples), 
                                'alpha_' + str(args.alpha),
                                'uncertainty_' + str(args.uncertainty),
                                'seed_' + str(args.seed))
        
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print("we are saving results to: ", save_dir)

    results = {"acc" : accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1}
    
    with open(os.path.join(save_dir, "results.json"), 'w') as f:
        json.dump(results, f)