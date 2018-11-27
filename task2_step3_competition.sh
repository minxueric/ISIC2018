#!/bin/bash

# loss
python ./task2_step3_competition.py -attribute globules -size 128 -loss_param dice -model vgg_unet -do_train -do_predict
python ./task2_step3_competition.py -attribute globules -size 128 -loss_param jacc -model vgg_unet -do_train -do_predict
python ./task2_step3_competition.py -attribute globules -size 128 -loss_param bce  -model vgg_unet -do_train -do_predict

# size
python ./task2_step3_competition.py -attribute globules -size 256 -loss_param dice -model vgg_unet -do_train -do_predict
python ./task2_step3_competition.py -attribute globules -size 256 -loss_param jacc -model vgg_unet -do_train -do_predict
python ./task2_step3_competition.py -attribute globules -size 256 -loss_param bce  -model vgg_unet -do_train -do_predict

# pre_proc
python ./task2_step3_competition.py -attribute globules -size 128 -loss_param dice -model vgg_unet -pre_proc -do_train -do_predict
python ./task2_step3_competition.py -attribute globules -size 128 -loss_param jacc -model vgg_unet -pre_proc -do_train -do_predict
python ./task2_step3_competition.py -attribute globules -size 128 -loss_param bce  -model vgg_unet -pre_proc -do_train -do_predict

python ./task2_step3_competition.py -attribute globules -size 256 -loss_param dice -model vgg_unet -pre_proc -do_train -do_predict
python ./task2_step3_competition.py -attribute globules -size 256 -loss_param jacc -model vgg_unet -pre_proc -do_train -do_predict
python ./task2_step3_competition.py -attribute globules -size 256 -loss_param bce  -model vgg_unet -pre_proc -do_train -do_predict


# loss
python ./task2_step3_competition.py -attribute globules -size 128 -loss_param dice -model vgg_unet -test_aug -do_train -do_predict
python ./task2_step3_competition.py -attribute globules -size 128 -loss_param jacc -model vgg_unet -test_aug -do_train -do_predict
python ./task2_step3_competition.py -attribute globules -size 128 -loss_param bce  -model vgg_unet -test_aug -do_train -do_predict

# size
python ./task2_step3_competition.py -attribute globules -size 256 -loss_param dice -model vgg_unet -test_aug -do_train -do_predict
python ./task2_step3_competition.py -attribute globules -size 256 -loss_param jacc -model vgg_unet -test_aug -do_train -do_predict
python ./task2_step3_competition.py -attribute globules -size 256 -loss_param bce  -model vgg_unet -test_aug -do_train -do_predict

# pre_proc
python ./task2_step3_competition.py -attribute globules -size 128 -loss_param dice -model vgg_unet -test_aug -pre_proc -do_train -do_predict
python ./task2_step3_competition.py -attribute globules -size 128 -loss_param jacc -model vgg_unet -test_aug -pre_proc -do_train -do_predict
python ./task2_step3_competition.py -attribute globules -size 128 -loss_param bce  -model vgg_unet -test_aug -pre_proc -do_train -do_predict

python ./task2_step3_competition.py -attribute globules -size 256 -loss_param dice -model vgg_unet -test_aug -pre_proc -do_train -do_predict
python ./task2_step3_competition.py -attribute globules -size 256 -loss_param jacc -model vgg_unet -test_aug -pre_proc -do_train -do_predict
python ./task2_step3_competition.py -attribute globules -size 256 -loss_param bce  -model vgg_unet -test_aug -pre_proc -do_train -do_predict


# unet
# loss
python ./task2_step3_competition.py -attribute globules -size 128 -loss_param dice -model unet -do_train -do_predict
python ./task2_step3_competition.py -attribute globules -size 128 -loss_param jacc -model unet -do_train -do_predict
python ./task2_step3_competition.py -attribute globules -size 128 -loss_param bce  -model unet -do_train -do_predict

# pre_proc
python ./task2_step3_competition.py -attribute globules -size 128 -loss_param dice -model unet -pre_proc -do_train -do_predict
python ./task2_step3_competition.py -attribute globules -size 128 -loss_param jacc -model unet -pre_proc -do_train -do_predict
python ./task2_step3_competition.py -attribute globules -size 128 -loss_param bce  -model unet -pre_proc -do_train -do_predict

# loss
python ./task2_step3_competition.py -attribute globules -size 128 -loss_param dice -model unet -test_aug -do_train -do_predict
python ./task2_step3_competition.py -attribute globules -size 128 -loss_param jacc -model unet -test_aug -do_train -do_predict
python ./task2_step3_competition.py -attribute globules -size 128 -loss_param bce  -model unet -test_aug -do_train -do_predict

# pre_proc
python ./task2_step3_competition.py -attribute globules -size 128 -loss_param dice -model unet -test_aug -pre_proc -do_train -do_predict
python ./task2_step3_competition.py -attribute globules -size 128 -loss_param jacc -model unet -test_aug -pre_proc -do_train -do_predict
python ./task2_step3_competition.py -attribute globules -size 128 -loss_param bce  -model unet -test_aug -pre_proc -do_train -do_predict

# unet2
# loss
python ./task2_step3_competition.py -attribute globules -size 128 -loss_param dice -model unet2 -do_train -do_predict
python ./task2_step3_competition.py -attribute globules -size 128 -loss_param jacc -model unet2 -do_train -do_predict
python ./task2_step3_competition.py -attribute globules -size 128 -loss_param bce  -model unet2 -do_train -do_predict

# pre_proc
python ./task2_step3_competition.py -attribute globules -size 128 -loss_param dice -model unet2 -pre_proc -do_train -do_predict
python ./task2_step3_competition.py -attribute globules -size 128 -loss_param jacc -model unet2 -pre_proc -do_train -do_predict
python ./task2_step3_competition.py -attribute globules -size 128 -loss_param bce  -model unet2 -pre_proc -do_train -do_predict

# loss
python ./task2_step3_competition.py -attribute globules -size 128 -loss_param dice -model unet2 -test_aug -do_train -do_predict
python ./task2_step3_competition.py -attribute globules -size 128 -loss_param jacc -model unet2 -test_aug -do_train -do_predict
python ./task2_step3_competition.py -attribute globules -size 128 -loss_param bce  -model unet2 -test_aug -do_train -do_predict

# pre_proc
python ./task2_step3_competition.py -attribute globules -size 128 -loss_param dice -model unet2 -test_aug -pre_proc -do_train -do_predict
python ./task2_step3_competition.py -attribute globules -size 128 -loss_param jacc -model unet2 -test_aug -pre_proc -do_train -do_predict
python ./task2_step3_competition.py -attribute globules -size 128 -loss_param bce  -model unet2 -test_aug -pre_proc -do_train -do_predict



