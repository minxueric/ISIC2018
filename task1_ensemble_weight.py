# -*- coding: utf-8 -*-

import os
import numpy as np
import pickle as pkl
import csv
from tqdm import tqdm, trange
import ISIC_dataset as ISIC

validation_base_path = './results/isic2018_validation_pkl/'
validation_folder = './datasets/ISIC2018_Task1-2_Validation_Input'
validation_predicted_folder = './results/ISIC2018_Validation_Predicted/ensemble_128_weighted_best10'

test_base_path = './results/isic2018_test_pkl/'
test_folder = './datasets/ISIC2018_Task1-2_Test_Input'
test_predicted_folder = './results/ISIC2018_Test_Predicted/ensemble_128_weigted_best10'

def ensemble(validation_base_path, validation_folder, validation_predicted_folder):
    pkl_files = []
    weights = []
    #weight_file = './file_weight_128.csv'
    #weight_file = './file_weight_128_144models.csv'
    #weight_file = './file_weight_128_144_updated models.csv'
    #weight_file = './file_weight_128_144_3rd version models.csv'
    weight_file = './10 best models weights for task 1.csv'
    #weight_file = './5 best models weights for task 1.csv'
    #weight_file = './20 best models weights for task 1.csv'

    with open(weight_file, 'rb') as f:
        rows = csv.reader(f, delimiter=',')
        #next(rows, None)
        for row in rows:
            if '.pkl' in row[0]:
                pkl_files.append(validation_base_path + row[0])
            else:
                pkl_files.append(validation_base_path + row[0] + '.pkl')
            weights.append(float(row[1]))

    print (len(pkl_files))
    print weights
    print np.sum(weights)

    mask_pred_challenge_list = []
    for i in trange(len(pkl_files)):
        mask_pred_challenge = pkl.load(open(pkl_files[i], 'rb'))
        mask_pred_challenge_list.append(mask_pred_challenge)
    mask_pred_challenge_list = np.array(mask_pred_challenge_list)
    print mask_pred_challenge_list.shape
    weights = np.array(weights)

    mask_pred_challenge = np.dot(mask_pred_challenge_list.transpose(1,2,3,0), weights)
    print mask_pred_challenge.shape

    if not os.path.exists(validation_predicted_folder):
        os.makedirs(validation_predicted_folder)

    cutoff = 0.5
    mask_pred_challenge_b = (np.where(mask_pred_challenge>=cutoff, 1, 0) * 255).astype(np.uint8)
    challenge_list = ISIC.list_from_folder(validation_folder)

    for i in trange(len(challenge_list)):
        _, _ = ISIC.show_images_full_sized(challenge_list,
                                           img_mask_pred_array=mask_pred_challenge_b,
                                           image_folder=validation_folder,
                                           mask_folder=None,
                                           index=i,
                                           output_folder=validation_predicted_folder,
                                           plot=False)

if __name__ == '__main__':
    ensemble(validation_base_path, validation_folder, validation_predicted_folder)
    ensemble(test_base_path, test_folder, test_predicted_folder)
