import numpy as np

def get_client_attributes(attribute_setting, face_seed):
    if attribute_setting == 'hair_vs_bald':
        client_attributes = [8,9,11,17,32,33]
        test_attribute = 4 # bald
        num_users = 6 
    
    elif attribute_setting == 'facial_features_vs_attactive':
        client_attributes = [4,6,7,13]
        test_attribute = 2
        num_users = 4
    
    elif attribute_setting == 'facial_features_vs_age':
        client_attributes = [3,4,22,24]
        test_attribute = 39
        num_users = 4
    
    # elif attribute_setting == 'random_5':
    #     np.random.seed(face_seed)
    #     client_attributes = np.random.choice([2,3,4,6,7,8,9,11,13,17,22,24,32,33], 5, replace=False)
    #     test_attribute = 39
    #     num_users = 5
    
    elif attribute_setting == 'random_5':
        np.random.seed(face_seed)
        client_attributes = np.random.choice([2,3,4,6,7,8,9,11,13,17,22,24,32,33,39], 5, replace=False)
        test_attribute = 31
        num_users = 5
    
    return client_attributes, test_attribute, num_users
        


        