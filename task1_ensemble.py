# -*- coding: utf-8 -*-

import os
import numpy as np
import pickle as pkl
import csv
from tqdm import tqdm, trange
import ISIC_dataset as ISIC

validation_base_path = './results/isic2018_validation_pkl/'
validation_folder = './datasets/ISIC2018_Task1-2_Validation_Input'
validation_predicted_folder = './results/ISIC2018_Validation_Predicted/ensemble_128_best10'

test_base_path = './results/isic2018_test_pkl/'
test_folder = './datasets/ISIC2018_Task1-2_Test_Input'
test_predicted_folder = './results/ISIC2018_Test_Predicted/ensemble_128_best10'


def ensemble(validation_base_path, validation_folder, validation_predicted_folder):
    pkl_files = []
    with open('./scores_task1.csv', 'rb') as f:
        rows = csv.reader(f, delimiter='\t')
        for row in rows:
            if '128' in row[0]:
                pkl_files.append(validation_base_path + row[0] + '.pkl')

    pkl_files = pkl_files[:10]
    print len(pkl_files)

    mask_pred_challenge_list = []
    for pkl_file in tqdm(pkl_files):
        try:
            mask_pred_challenge = pkl.load(open(pkl_file, 'rb'))
            if mask_pred_challenge.max() < 0.5:
                print 'garbage model'
                continue
            mask_pred_challenge_list.append(mask_pred_challenge)
        except:
            print ( pkl_file, 'Not found')
    print 'Finally use {} models'.format(len(mask_pred_challenge_list))
    mask_pred_challenge_list = np.array(mask_pred_challenge_list)
    mask_pred_challenge = np.mean(mask_pred_challenge_list, axis=0)

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
