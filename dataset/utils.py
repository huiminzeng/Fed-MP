import os
import glob
import numpy as np

from .attributes_dict import *

def collate_fn(data):
    image_set, targets = zip(*data)
    return list(image_set), list(targets)

def caltech_split_clients(data_dir, num_samples, num_users):
    """ Caltech101 and Food101 use this same function to create federated clients.
    """
    dict_users = {}
    
    all_categories = sorted(os.listdir(data_dir))
    classes_per_client = len(all_categories) // (num_users + 1)
    # client data
    for i in range(num_users):
        dict_users[i] = []
        np.random.seed(i+1)
        selected_y = np.random.choice(all_categories, classes_per_client, replace=False)
        all_categories = set(all_categories) - set(selected_y)
        all_categories = sorted(list(all_categories))

        training_samples = []
        training_classes = []
        for current_y in selected_y:
            samples = np.random.choice((glob.glob(data_dir + '/' + current_y + '/*.jpg')), num_samples, replace=False)
            for sample in samples:
                training_samples.append(sample)
            training_classes.append(current_y)
        
        dict_users[i] = [training_classes, training_samples]

    # test data
    test_data = []
    test_classes = []
    for current_y in all_categories:
        samples =glob.glob(data_dir + '/' + current_y + '/*.jpg')
        for sample in samples:
            test_data.append(sample)
        test_classes.append(current_y)

    return dict_users, classes_per_client, test_data, test_classes

def fgvc_split_clients(data_dir, id2label_train, label2id_train, id2label_test, label2id_test, num_samples, num_users):
    dict_users = {}
    
    all_categories = list(label2id_train.keys())
    all_categories = sorted(all_categories)
    classes_per_client = len(all_categories) // (num_users + 1)
    # client data
    for i in range(num_users):
        dict_users[i] = []
        np.random.seed(i)
        selected_y = np.random.choice(all_categories, classes_per_client, replace=False)
        all_categories = set(all_categories) - set(selected_y)
        all_categories = sorted(list(all_categories))

        training_samples = []
        training_classes = []
        for current_y in selected_y:
            samples = np.random.choice(label2id_train[current_y], num_samples, replace=False)
            for sample in samples:
                training_samples.append(os.path.join(data_dir, sample + '.jpg'))
            training_classes.append(current_y)
        
        dict_users[i] = [training_classes, training_samples, id2label_train]
        
    # test data
    test_data = []
    test_classes = []
    for current_y in all_categories:
        samples = label2id_test[current_y]
        for sample in samples:
            test_data.append(os.path.join(data_dir, sample + '.jpg'))
        test_classes.append(current_y)

    return dict_users, classes_per_client, test_data, test_classes

def flower_split_clients(data_root, num_samples, num_users):
    dict_users = {}
    
    # split train data
    data_dir = os.path.join(data_root, 'train')
    all_categories = list(range(1,103))
    classes_per_client = len(all_categories) // (num_users + 1)
    # client data
    for i in range(num_users):
        dict_users[i] = []
        np.random.seed(i+1)
        selected_y = np.random.choice(all_categories, classes_per_client, replace=False)
        all_categories = set(all_categories) - set(selected_y)
        all_categories = sorted(list(all_categories))

        training_samples = []
        training_classes = []
        for current_y in selected_y:
            samples = np.random.choice((glob.glob(data_dir + '/' + str(current_y) + '/*.jpg')), num_samples, replace=False)
            for sample in samples:
                training_samples.append(sample)
            training_classes.append(flower_attribute_dict[str(current_y)])
        
        dict_users[i] = [training_classes, training_samples]

    # test data
    test_data = []
    test_classes = []
    for current_y in all_categories:
        samples =glob.glob(data_dir + '/' + str(current_y) + '/*.jpg')
        for sample in samples:
            test_data.append(sample)
        test_classes.append(flower_attribute_dict[str(current_y)])

    return dict_users, classes_per_client, test_data, test_classes

def ucf_split_clients(data_dir, num_samples, num_users):
    dict_users = {}
    import re  
    regex = r"[A-Z][a-z]*[^A-Z]"

    # split train data
    all_categories = sorted(os.listdir(data_dir))
    classes_per_client = len(all_categories) // (num_users + 1)
    # client data
    for i in range(num_users):
        dict_users[i] = []
        np.random.seed(i)
        selected_y = np.random.choice(all_categories, classes_per_client, replace=False)
        all_categories = set(all_categories) - set(selected_y)
        all_categories = sorted(list(all_categories))
        training_samples = []
        training_classes = []
        for current_y in selected_y:
            try:
                samples = np.random.choice((glob.glob(data_dir + '/' + str(current_y) + '/*.jpg')), num_samples, replace=False)
            except:
                samples = glob.glob(data_dir + '/' + str(current_y) + '/*.jpg')

            for sample in samples:
                training_samples.append(sample)
            
            match = re.findall(regex, current_y) 
            training_classes.append(' '.join(match))

        dict_users[i] = [training_classes, training_samples]

    # test data
    test_data = []
    test_classes = []
    for current_y in all_categories:
        samples =glob.glob(data_dir + '/' + current_y + '/*.jpg')
        for sample in samples:
            test_data.append(sample)
        
        match = re.findall(regex, current_y) 
        test_classes.append(' '.join(match))

    return dict_users, classes_per_client, test_data, test_classes

def cars_split_clients(data_dir, id2label_train, label2id_train, id2label_test, label2id_test, num_samples, num_users):
    dict_users = {}
    
    all_categories = list(label2id_train.keys())
    all_categories = sorted(all_categories)
    classes_per_client = len(all_categories) // (num_users + 1)
    # client data
    for i in range(num_users):
        dict_users[i] = []
        np.random.seed(i+2)
        selected_y = np.random.choice(all_categories, classes_per_client, replace=False)
        all_categories = set(all_categories) - set(selected_y)
        all_categories = sorted(list(all_categories))

        training_samples = []
        training_classes = []
        for current_y in selected_y:
            samples = np.random.choice(label2id_train[current_y], num_samples, replace=False)
            for sample in samples:
                training_samples.append(os.path.join(data_dir, 'cars_train', 'cars_train', sample + '.jpg'))
            training_classes.append(current_y)
        
        dict_users[i] = [training_classes, training_samples, id2label_train]
        
    # test data
    test_data = []
    test_classes = []
    for current_y in all_categories:
        samples = label2id_test[current_y]
        for sample in samples:
            test_data.append(os.path.join(data_dir, 'cars_test', 'cars_test', sample + '.jpg'))
        test_classes.append(current_y)

    return dict_users, classes_per_client, test_data, test_classes
