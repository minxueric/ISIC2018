#!/bin/bash

# loss and model
python ./task1_competition.py -loss_param dice -model vgg_unet     
python ./task1_competition.py -loss_param dice -model unet         
python ./task1_competition.py -loss_param dice -model unet2        
python ./task1_competition.py -loss_param jacc -model vgg_unet     
python ./task1_competition.py -loss_param jacc -model unet         
python ./task1_competition.py -loss_param jacc -model unet2        
python ./task1_competition.py -loss_param bce  -model vgg_unet     
python ./task1_competition.py -loss_param bce  -model unet         
python ./task1_competition.py -loss_param bce  -model unet2        

# use archive
python ./task1_competition.py -loss_param dice -model vgg_unet -use_archive     
python ./task1_competition.py -loss_param dice -model unet     -use_archive     
python ./task1_competition.py -loss_param dice -model unet2    -use_archive     
python ./task1_competition.py -loss_param jacc -model vgg_unet -use_archive     
python ./task1_competition.py -loss_param jacc -model unet     -use_archive     
python ./task1_competition.py -loss_param jacc -model unet2    -use_archive     
python ./task1_competition.py -loss_param bce  -model vgg_unet -use_archive     
python ./task1_competition.py -loss_param bce  -model unet     -use_archive     
python ./task1_competition.py -loss_param bce  -model unet2    -use_archive     

## task1-0
# use ph2
python ./task1_competition.py -loss_param dice -model vgg_unet -use_ph2     
python ./task1_competition.py -loss_param dice -model unet     -use_ph2     
python ./task1_competition.py -loss_param dice -model unet2    -use_ph2     
python ./task1_competition.py -loss_param jacc -model vgg_unet -use_ph2     
python ./task1_competition.py -loss_param jacc -model unet     -use_ph2     
python ./task1_competition.py -loss_param jacc -model unet2    -use_ph2     
python ./task1_competition.py -loss_param bce  -model vgg_unet -use_ph2     
python ./task1_competition.py -loss_param bce  -model unet     -use_ph2     
python ./task1_competition.py -loss_param bce  -model unet2    -use_ph2     

# use both ph2 and archive
python ./task1_competition.py -loss_param dice -model vgg_unet -use_archive -use_ph2     
python ./task1_competition.py -loss_param dice -model unet     -use_archive -use_ph2     
python ./task1_competition.py -loss_param dice -model unet2    -use_archive -use_ph2     
python ./task1_competition.py -loss_param jacc -model vgg_unet -use_archive -use_ph2     
python ./task1_competition.py -loss_param jacc -model unet     -use_archive -use_ph2     
python ./task1_competition.py -loss_param jacc -model unet2    -use_archive -use_ph2     
python ./task1_competition.py -loss_param bce  -model vgg_unet -use_archive -use_ph2     
python ./task1_competition.py -loss_param bce  -model unet     -use_archive -use_ph2     
python ./task1_competition.py -loss_param bce  -model unet2    -use_archive -use_ph2     


# use pre_process
python ./task1_competition.py -loss_param dice -model vgg_unet -pre_proc     
python ./task1_competition.py -loss_param dice -model unet     -pre_proc     
python ./task1_competition.py -loss_param dice -model unet2    -pre_proc     
python ./task1_competition.py -loss_param jacc -model vgg_unet -pre_proc     
python ./task1_competition.py -loss_param jacc -model unet     -pre_proc     
python ./task1_competition.py -loss_param jacc -model unet2    -pre_proc     
python ./task1_competition.py -loss_param bce  -model vgg_unet -pre_proc     
python ./task1_competition.py -loss_param bce  -model unet     -pre_proc     
python ./task1_competition.py -loss_param bce  -model unet2    -pre_proc     

# use archive
python ./task1_competition.py -loss_param dice -model vgg_unet -pre_proc -use_archive     
python ./task1_competition.py -loss_param dice -model unet     -pre_proc -use_archive     
python ./task1_competition.py -loss_param dice -model unet2    -pre_proc -use_archive     
python ./task1_competition.py -loss_param jacc -model vgg_unet -pre_proc -use_archive     
python ./task1_competition.py -loss_param jacc -model unet     -pre_proc -use_archive     
python ./task1_competition.py -loss_param jacc -model unet2    -pre_proc -use_archive     
python ./task1_competition.py -loss_param bce  -model vgg_unet -pre_proc -use_archive     
python ./task1_competition.py -loss_param bce  -model unet     -pre_proc -use_archive     
python ./task1_competition.py -loss_param bce  -model unet2    -pre_proc -use_archive     

# use ph2
python ./task1_competition.py -loss_param dice -model vgg_unet -pre_proc -use_ph2     
python ./task1_competition.py -loss_param dice -model unet     -pre_proc -use_ph2     
python ./task1_competition.py -loss_param dice -model unet2    -pre_proc -use_ph2     
python ./task1_competition.py -loss_param jacc -model vgg_unet -pre_proc -use_ph2     
python ./task1_competition.py -loss_param jacc -model unet     -pre_proc -use_ph2     
python ./task1_competition.py -loss_param jacc -model unet2    -pre_proc -use_ph2     
python ./task1_competition.py -loss_param bce  -model vgg_unet -pre_proc -use_ph2     
python ./task1_competition.py -loss_param bce  -model unet     -pre_proc -use_ph2     
python ./task1_competition.py -loss_param bce  -model unet2    -pre_proc -use_ph2     

# use both ph2 and archive
python ./task1_competition.py -loss_param dice -model vgg_unet -pre_proc -use_archive -use_ph2     
python ./task1_competition.py -loss_param dice -model unet     -pre_proc -use_archive -use_ph2     
python ./task1_competition.py -loss_param dice -model unet2    -pre_proc -use_archive -use_ph2     
python ./task1_competition.py -loss_param jacc -model vgg_unet -pre_proc -use_archive -use_ph2     
python ./task1_competition.py -loss_param jacc -model unet     -pre_proc -use_archive -use_ph2     
python ./task1_competition.py -loss_param jacc -model unet2    -pre_proc -use_archive -use_ph2     
python ./task1_competition.py -loss_param bce  -model vgg_unet -pre_proc -use_archive -use_ph2     
python ./task1_competition.py -loss_param bce  -model unet     -pre_proc -use_archive -use_ph2     
python ./task1_competition.py -loss_param bce  -model unet2    -pre_proc -use_archive -use_ph2     


# use size 256
python ./task1_competition.py -size 256 -loss_param dice -model vgg_unet     
python ./task1_competition.py -size 256 -loss_param jacc -model vgg_unet     
python ./task1_competition.py -size 256 -loss_param bce  -model vgg_unet     
# use archive
python ./task1_competition.py -size 256 -loss_param dice -model vgg_unet -use_archive     
python ./task1_competition.py -size 256 -loss_param jacc -model vgg_unet -use_archive     
python ./task1_competition.py -size 256 -loss_param bce  -model vgg_unet -use_archive     
# use ph2
python ./task1_competition.py -size 256 -loss_param dice -model vgg_unet -use_ph2     
python ./task1_competition.py -size 256 -loss_param jacc -model vgg_unet -use_ph2     
python ./task1_competition.py -size 256 -loss_param bce  -model vgg_unet -use_ph2     
# use both ph2 and archive
python ./task1_competition.py -size 256 -loss_param dice -model vgg_unet -use_archive -use_ph2     
python ./task1_competition.py -size 256 -loss_param jacc -model vgg_unet -use_archive -use_ph2     
python ./task1_competition.py -size 256 -loss_param bce  -model vgg_unet -use_archive -use_ph2     

# use pre_process
python ./task1_competition.py -size 256 -loss_param dice -model vgg_unet -pre_proc     
python ./task1_competition.py -size 256 -loss_param jacc -model vgg_unet -pre_proc     
python ./task1_competition.py -size 256 -loss_param bce  -model vgg_unet -pre_proc     
# use archive
python ./task1_competition.py -size 256 -loss_param dice -model vgg_unet -pre_proc -use_archive     
python ./task1_competition.py -size 256 -loss_param jacc -model vgg_unet -pre_proc -use_archive     
python ./task1_competition.py -size 256 -loss_param bce  -model vgg_unet -pre_proc -use_archive     
# use ph2
python ./task1_competition.py -size 256 -loss_param dice -model vgg_unet -pre_proc -use_ph2     
python ./task1_competition.py -size 256 -loss_param jacc -model vgg_unet -pre_proc -use_ph2     
python ./task1_competition.py -size 256 -loss_param bce  -model vgg_unet -pre_proc -use_ph2     
# use both ph2 and archive
python ./task1_competition.py -size 256 -loss_param dice -model vgg_unet -pre_proc -use_archive -use_ph2     
python ./task1_competition.py -size 256 -loss_param jacc -model vgg_unet -pre_proc -use_archive -use_ph2     
python ./task1_competition.py -size 256 -loss_param bce  -model vgg_unet -pre_proc -use_archive -use_ph2     


 # do predict without training
# loss and model
python ./task1_competition.py -loss_param dice -model vgg_unet -test_aug  
python ./task1_competition.py -loss_param dice -model unet     -test_aug  
python ./task1_competition.py -loss_param dice -model unet2    -test_aug  
python ./task1_competition.py -loss_param jacc -model vgg_unet -test_aug  
python ./task1_competition.py -loss_param jacc -model unet     -test_aug  
python ./task1_competition.py -loss_param jacc -model unet2    -test_aug  
python ./task1_competition.py -loss_param bce  -model vgg_unet -test_aug  
python ./task1_competition.py -loss_param bce  -model unet     -test_aug  
python ./task1_competition.py -loss_param bce  -model unet2    -test_aug  

# use archive
python ./task1_competition.py -loss_param dice -model vgg_unet -use_archive -test_aug  
python ./task1_competition.py -loss_param dice -model unet     -use_archive -test_aug  
python ./task1_competition.py -loss_param dice -model unet2    -use_archive -test_aug  
python ./task1_competition.py -loss_param jacc -model vgg_unet -use_archive -test_aug  
python ./task1_competition.py -loss_param jacc -model unet     -use_archive -test_aug  
python ./task1_competition.py -loss_param jacc -model unet2    -use_archive -test_aug  
python ./task1_competition.py -loss_param bce  -model vgg_unet -use_archive -test_aug  
python ./task1_competition.py -loss_param bce  -model unet     -use_archive -test_aug  
python ./task1_competition.py -loss_param bce  -model unet2    -use_archive -test_aug  


# use ph2
python ./task1_competition.py -loss_param dice -model vgg_unet -use_ph2 -test_aug  
python ./task1_competition.py -loss_param dice -model unet     -use_ph2 -test_aug  
python ./task1_competition.py -loss_param dice -model unet2    -use_ph2 -test_aug  
python ./task1_competition.py -loss_param jacc -model vgg_unet -use_ph2 -test_aug  
python ./task1_competition.py -loss_param jacc -model unet     -use_ph2 -test_aug  
python ./task1_competition.py -loss_param jacc -model unet2    -use_ph2 -test_aug  
python ./task1_competition.py -loss_param bce  -model vgg_unet -use_ph2 -test_aug  
python ./task1_competition.py -loss_param bce  -model unet     -use_ph2 -test_aug  
python ./task1_competition.py -loss_param bce  -model unet2    -use_ph2 -test_aug  

# use both ph2 and archive
python ./task1_competition.py -loss_param dice -model vgg_unet -use_archive -use_ph2 -test_aug  
python ./task1_competition.py -loss_param dice -model unet     -use_archive -use_ph2 -test_aug  
python ./task1_competition.py -loss_param dice -model unet2    -use_archive -use_ph2 -test_aug  
python ./task1_competition.py -loss_param jacc -model vgg_unet -use_archive -use_ph2 -test_aug  
python ./task1_competition.py -loss_param jacc -model unet     -use_archive -use_ph2 -test_aug  
python ./task1_competition.py -loss_param jacc -model unet2    -use_archive -use_ph2 -test_aug  
python ./task1_competition.py -loss_param bce  -model vgg_unet -use_archive -use_ph2 -test_aug  
python ./task1_competition.py -loss_param bce  -model unet     -use_archive -use_ph2 -test_aug  
python ./task1_competition.py -loss_param bce  -model unet2    -use_archive -use_ph2 -test_aug  



# use pre_process
python ./task1_competition.py -loss_param dice -model vgg_unet -pre_proc -test_aug  
python ./task1_competition.py -loss_param dice -model unet     -pre_proc -test_aug  
python ./task1_competition.py -loss_param dice -model unet2    -pre_proc -test_aug  
python ./task1_competition.py -loss_param jacc -model vgg_unet -pre_proc -test_aug  
python ./task1_competition.py -loss_param jacc -model unet     -pre_proc -test_aug  
python ./task1_competition.py -loss_param jacc -model unet2    -pre_proc -test_aug  
python ./task1_competition.py -loss_param bce  -model vgg_unet -pre_proc -test_aug  
python ./task1_competition.py -loss_param bce  -model unet     -pre_proc -test_aug  
python ./task1_competition.py -loss_param bce  -model unet2    -pre_proc -test_aug  

# use archive
python ./task1_competition.py -loss_param dice -model vgg_unet -pre_proc -use_archive -test_aug  
python ./task1_competition.py -loss_param dice -model unet     -pre_proc -use_archive -test_aug  
python ./task1_competition.py -loss_param dice -model unet2    -pre_proc -use_archive -test_aug  
python ./task1_competition.py -loss_param jacc -model vgg_unet -pre_proc -use_archive -test_aug  
python ./task1_competition.py -loss_param jacc -model unet     -pre_proc -use_archive -test_aug  
python ./task1_competition.py -loss_param jacc -model unet2    -pre_proc -use_archive -test_aug  
python ./task1_competition.py -loss_param bce  -model vgg_unet -pre_proc -use_archive -test_aug  
python ./task1_competition.py -loss_param bce  -model unet     -pre_proc -use_archive -test_aug  
python ./task1_competition.py -loss_param bce  -model unet2    -pre_proc -use_archive -test_aug  

# use ph2
python ./task1_competition.py -loss_param dice -model vgg_unet -pre_proc -use_ph2 -test_aug  
python ./task1_competition.py -loss_param dice -model unet     -pre_proc -use_ph2 -test_aug  
python ./task1_competition.py -loss_param dice -model unet2    -pre_proc -use_ph2 -test_aug  
python ./task1_competition.py -loss_param jacc -model vgg_unet -pre_proc -use_ph2 -test_aug  
python ./task1_competition.py -loss_param jacc -model unet     -pre_proc -use_ph2 -test_aug  
python ./task1_competition.py -loss_param jacc -model unet2    -pre_proc -use_ph2 -test_aug  
python ./task1_competition.py -loss_param bce  -model vgg_unet -pre_proc -use_ph2 -test_aug  
python ./task1_competition.py -loss_param bce  -model unet     -pre_proc -use_ph2 -test_aug  
python ./task1_competition.py -loss_param bce  -model unet2    -pre_proc -use_ph2 -test_aug  

# use both ph2 and archive
python ./task1_competition.py -loss_param dice -model vgg_unet -pre_proc -use_archive -use_ph2 -test_aug  
python ./task1_competition.py -loss_param dice -model unet     -pre_proc -use_archive -use_ph2 -test_aug  
python ./task1_competition.py -loss_param dice -model unet2    -pre_proc -use_archive -use_ph2 -test_aug  
python ./task1_competition.py -loss_param jacc -model vgg_unet -pre_proc -use_archive -use_ph2 -test_aug  
python ./task1_competition.py -loss_param jacc -model unet     -pre_proc -use_archive -use_ph2 -test_aug  
python ./task1_competition.py -loss_param jacc -model unet2    -pre_proc -use_archive -use_ph2 -test_aug  
python ./task1_competition.py -loss_param bce  -model vgg_unet -pre_proc -use_archive -use_ph2 -test_aug  
python ./task1_competition.py -loss_param bce  -model unet     -pre_proc -use_archive -use_ph2 -test_aug  
python ./task1_competition.py -loss_param bce  -model unet2    -pre_proc -use_archive -use_ph2 -test_aug  


# use size 256
python ./task1_competition.py -size 256 -loss_param dice -model vgg_unet -test_aug  
python ./task1_competition.py -size 256 -loss_param jacc -model vgg_unet -test_aug  
python ./task1_competition.py -size 256 -loss_param bce  -model vgg_unet -test_aug  
# use archive
python ./task1_competition.py -size 256 -loss_param dice -model vgg_unet -use_archive -test_aug  
python ./task1_competition.py -size 256 -loss_param jacc -model vgg_unet -use_archive -test_aug  
python ./task1_competition.py -size 256 -loss_param bce  -model vgg_unet -use_archive -test_aug  
# use ph2
python ./task1_competition.py -size 256 -loss_param dice -model vgg_unet -use_ph2 -test_aug  
python ./task1_competition.py -size 256 -loss_param jacc -model vgg_unet -use_ph2 -test_aug  
python ./task1_competition.py -size 256 -loss_param bce  -model vgg_unet -use_ph2 -test_aug  
# use both ph2 and archive
python ./task1_competition.py -size 256 -loss_param dice -model vgg_unet -use_archive -use_ph2 -test_aug  
python ./task1_competition.py -size 256 -loss_param jacc -model vgg_unet -use_archive -use_ph2 -test_aug  
python ./task1_competition.py -size 256 -loss_param bce  -model vgg_unet -use_archive -use_ph2 -test_aug  

# use pre_process
python ./task1_competition.py -size 256 -loss_param dice -model vgg_unet -pre_proc -test_aug  
python ./task1_competition.py -size 256 -loss_param jacc -model vgg_unet -pre_proc -test_aug  
python ./task1_competition.py -size 256 -loss_param bce  -model vgg_unet -pre_proc -test_aug  
# use archive
python ./task1_competition.py -size 256 -loss_param dice -model vgg_unet -pre_proc -use_archive -test_aug  
python ./task1_competition.py -size 256 -loss_param jacc -model vgg_unet -pre_proc -use_archive -test_aug  
python ./task1_competition.py -size 256 -loss_param bce  -model vgg_unet -pre_proc -use_archive -test_aug  
# use ph2
python ./task1_competition.py -size 256 -loss_param dice -model vgg_unet -pre_proc -use_ph2 -test_aug  
python ./task1_competition.py -size 256 -loss_param jacc -model vgg_unet -pre_proc -use_ph2 -test_aug  
python ./task1_competition.py -size 256 -loss_param bce  -model vgg_unet -pre_proc -use_ph2 -test_aug  
# use both ph2 and archive
python ./task1_competition.py -size 256 -loss_param dice -model vgg_unet -pre_proc -use_archive -use_ph2 -test_aug  
python ./task1_competition.py -size 256 -loss_param jacc -model vgg_unet -pre_proc -use_archive -use_ph2 -test_aug  
python ./task1_competition.py -size 256 -loss_param bce  -model vgg_unet -pre_proc -use_archive -use_ph2 -test_aug  



