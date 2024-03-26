# Open-Vocabulary Federated Learning with Multimodal Prototyping

This repository is the PyTorch impelementation for the paper "Open-Vocabulary Federated Learning with Multimodal Prototyping".

<img src=media/1.jpg width=400>

Existing federated learning (FL) studies usually assume the training label space and test label space are identical. However, in real-world applications, this assumption is too ideal to be true. A new user could come up with queries that involve data from unseen classes, and such open-vocabulary queries would directly defect such traditional FL systems. Therefore, in this work, we explicitly focus on the under-explored open-vocabulary challenge in FL. That is, for a new user, the global server shall understand her/his query that involves arbitrary unknown classes. To address this problem, we present a novel adaptation framework tailored for VLMs in the context of FL, named as Federated Multimodal Prototyping (Fed-MP). Fed-MP adaptively aggregates the local model weights based on light-weight client residuals, and makes predictions based on a novel multimodal prototyping mechanism.

<img src=media/2.jpg>

<!-- ## Citing 

Please consider citing the following paper if you use our methods in your research:
```
@inproceedings{zeng2022attacking,
  title={On Attacking Out-Domain Uncertainty Estimation in Deep Neural Networks},
  author={Zeng, Huimin and Yue, Zhenrui and Zhang, Yang and Kou, Ziyi and Shang, Lanyu and Wang, Dong},
  year={2022},
  organization={IJCAI}
}
``` -->

## Requirements

For our running environment see requirements.txt

## Datasets
- The datasets used in our work are all publicly available. They could be found and download from their official websites. After downloading, change the '--data_dir' argument in 'train.py' and 'eval.py'.

- Example folder structure
```
    ├── ...
    ├── data                   
    │   ├── Caltech101 
    │   ├── UCF101
    │   ├── Food101
    │   └── ...
    ├── nets
    │   └── ...
    ├── utils
    │   └── ...
    ├── train.py
    └── ...
```

## Scripts

- Stage 1: train adapter
   - Example
       ```
       python3 train.py --dataset --num_users --num_samples --alpha --seed 
    
       ```
   - Hyperparameters
      ```
      --dataset                 # select from 'caltech101', 'ucf101', 'food101', 'flower102', 'fgvc', 'cars'
      --num_clients             # number of federated clients
      --num_samples             # number of training users per training class
      --alpha                   # rescaling client residuals
      --seed                    # random seeds
      ```
    - After executing the script, the trained global adapter will be automatically saved to a new folder `trained_models`.

- Stage 2: perform multimodal prototyping and evaluation
    - Example
       ```
       python3 eval.py --dataset --num_users --num_samples --alpha --uncertainty --seed
       ```
   - Hyperparameters
      ```
      --dataset                 # select from 'caltech101', 'ucf101', 'food101', 'flower102', 'fgvc', 'cars'
      --num_clients             # number of federated clients
      --num_samples             # number of training users per training class
      --alpha                   # rescaling client residuals
      --uncertainty             # confidence threshold, range from 0.0 - 1.0
      --seed                    # random seeds
      ```
    - After executing the script, the multimodal prototyping will be performed and the evaluation results will be saved to a new folder `experiments`.


## Performance

The table below reports our main performance results, with best results marked in bold and second best results underlined. For training and evaluation details, please refer to our paper.

<img src=media/results.jpg>
