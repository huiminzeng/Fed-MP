import os
import scipy.io as sio

from .utils import *
from .datasets import *

def get_dataset(args, process=None):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """
    dataset = args.dataset
    num_users = args.num_users
    num_samples = args.num_samples
    
    if dataset == 'caltech101':
        data_dir = args.data_dir + "/Caltech101/caltech-101/101_ObjectCategories"
        user_groups, classes_per_client, test_data_path, test_labels = caltech_split_clients(data_dir, num_samples, num_users)

        test_dataset = myCaltech(data_path=test_data_path,
                                all_classes = test_labels,
                                process=process)
    elif dataset == 'food101':
        data_dir = args.data_dir + "/Food101/food-101/images"
        user_groups, classes_per_client, test_data_path, test_labels  = caltech_split_clients(data_dir, num_samples, num_users)
        test_dataset = myCaltech(data_path=test_data_path,
                                all_classes = test_labels,
                                process=process)
        
    elif dataset == 'flower102':
        data_dir = args.data_dir + "/Flowers102/flower_data"
        user_groups,classes_per_client, test_data_path, test_labels = flower_split_clients(data_dir, num_samples, num_users)

        test_dataset = myFlower(data_path=test_data_path,
                                all_classes = test_labels,
                                process=process)
    
    elif dataset == 'ucf101':
        data_dir = args.data_dir + "/UCF101/images"
        user_groups, classes_per_client, test_data_path, test_labels = ucf_split_clients(data_dir, num_samples, num_users)

        test_dataset = myUCF(data_path=test_data_path,
                            all_classes = test_labels,
                            process=process)

    elif dataset == 'fgvc':
        data_dir = args.data_dir + "/FGVC_Aircraft/images_processed"
        label_dir = args.data_dir + "/FGVC_Aircraft/fgvc-aircraft-2013b/data"

        # get train data and train labels
        id2label_train = {} # image id to labels (classes)
        label2id_train = {}
        with open(os.path.join(label_dir, 'images_family_train.txt')) as f:
            lines = f.readlines()
            for line in lines:
                line_split = line.strip()
                image_id = line_split[:7]
                label = line_split[8:]
                id2label_train[image_id] = label

                if label not in label2id_train:
                    label2id_train[label] = [image_id]
                else:
                    label2id_train[label].append(image_id)
                    

        # get test data and test labels
        id2label_test = {} # image id to labels (classes)
        label2id_test = {}
        with open(os.path.join(label_dir, 'images_family_test.txt')) as f:
            lines = f.readlines()
            for line in lines:
                line_split = line.strip()
                image_id = line_split[:7]
                label = line_split[8:]
                id2label_test[image_id] = label

                if label not in label2id_test:
                    label2id_test[label] = [image_id]
                else:
                    label2id_test[label].append(image_id)

        user_groups, classes_per_client, test_data_path, test_labels = fgvc_split_clients(data_dir, id2label_train, label2id_train, 
                                                                                        id2label_test, label2id_test, num_samples, num_users)
        
        test_dataset = myFGVC(data_path=test_data_path,
                            all_classes = test_labels,
                            label_dict = id2label_test,
                            process=process)

    elif dataset == 'cars':
        data_dir = args.data_dir + '/StanfordCars/'
        # get class names:
        file_name = args.data_dir + '/StanfordCars/cars_annos.mat'
        annos_mat =  sio.loadmat(file_name)
        class_list = [arr[0] for arr in annos_mat['class_names'][0]]
        
        # process training labels
        train_label_file = args.data_dir + '/StanfordCars/cars_train_annos.mat'
        train_annos_mat =  sio.loadmat(train_label_file)

        id2label_train = {} # image id to labels (classes)
        label2id_train = {}

        for i in range(train_annos_mat['annotations'][0].shape[0]):
            image_id = train_annos_mat['annotations'][0][i][5][0].split('.')[0]
            label = train_annos_mat['annotations'][0][i][4][0][0] - 1
            id2label_train[image_id] = class_list[label]

            if class_list[label] not in label2id_train:
                label2id_train[class_list[label]] = [image_id]
            else:
                label2id_train[class_list[label]].append(image_id)
        
        # process test labels
        test_label_file = args.data_dir + '/StanfordCars/cars_test_annos_withlabels.mat'
        test_annos_mat =  sio.loadmat(test_label_file)

        id2label_test = {} # image id to labels (classes)
        label2id_test = {}

        for i in range(test_annos_mat['annotations'][0].shape[0]):
            image_id = test_annos_mat['annotations'][0][i][5][0].split('.')[0]
            label = test_annos_mat['annotations'][0][i][4][0][0] - 1
            id2label_test[image_id] = class_list[label]

            if class_list[label] not in label2id_test:
                label2id_test[class_list[label]] = [image_id]
            else:
                label2id_test[class_list[label]].append(image_id)

        user_groups, classes_per_client, test_data_path, test_labels = cars_split_clients(data_dir, id2label_train, label2id_train, 
                                                                                        id2label_test, label2id_test, num_samples, num_users)
        
        test_dataset = myFGVC(data_path=test_data_path,
                            all_classes = test_labels,
                            label_dict = id2label_test,
                            process=process)
        
    return user_groups, classes_per_client, test_dataset
