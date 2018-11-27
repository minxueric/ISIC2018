import os
import numpy as np
import pandas as pd
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import pickle as pkl
import ISIC_dataset as ISIC
from metrics import dice_loss, jacc_loss, jacc_coef, dice_jacc_mean, jacc_coef_th, jacc_loss_th
import models
from sklearn.model_selection import train_test_split
from tqdm import tqdm, trange
from pre_processing import my_PreProc
from imgaug_test_cc import transform_img, reverse_gt
import argparse

np.random.seed(4)
K.set_image_dim_ordering('th')  # Theano dimension ordering: (channels, width, height)
                                # some changes will be necessary to run with tensorflow

# Folder of training images and masks
training_folder = "./datasets/ISIC2018_Task1-2_Training_Input"
training_mask_folder = "./datasets/ISIC2018_Task1_Training_GroundTruth"

# Folder of validation and test images
val_folder = "./datasets/ISIC2018_Task1-2_Validation_Input"
test_folder = "./datasets/ISIC2018_Task1-2_Test_Input"

# Folder to store predicted masks
val_predicted_folder = "./results/ISIC2018_Validation_Predicted"
test_predicted_folder = "./results/ISIC2018_Test_Predicted"

# Folder of external data
# ISIC archive data
archive_folder = "./datasets/Archive/isic_archive_img_seg/image"
archive_mask_folder = './datasets/Archive/isic_archive_img_seg/segmentation'
archive_img_seg_map = './datasets/Archive/isic_archive_img_seg/img_seg_count.csv'
# PH2 data
ph2_folder = "./datasets/PH2 Dataset images"

loss_options = {'bce': 'binary_crossentropy',
                'dice': dice_loss,
                'jacc': jacc_loss,
                'mse': 'mean_squared_error',
                'jacc_th': jacc_loss_th}

parser = argparse.ArgumentParser(description='Process U-net training testing arguments')
parser.add_argument('-size', dest='size', type=int, default=128, help='size of input images')
parser.add_argument('-loss_param', dest='loss_param', type=str, default='dice', help='loss function of Unet')
parser.add_argument('-model', dest='model', type=str, default='vgg_unet', help='name of model architecture')
parser.add_argument('-pre_proc', dest='pre_proc', action='store_true', default=False, help='whether using preprocessing')
parser.add_argument('-test_aug', dest='test_aug', action='store_true', default=False, help='whether using test data augmentation')
parser.add_argument('-use_archive', dest='use_archive', action='store_true', default=False, help='whether using extra ISIC archive data')
parser.add_argument('-use_ph2', dest='use_ph2', action='store_true', default=False, help='whether using extra PH2 data')

parser.add_argument('-do_train', dest='do_train', action='store_true', default=False, help='do training step')
parser.add_argument('-do_evaluate', dest='do_evaluate', action='store_true', default=False, help='do evaluation step')
parser.add_argument('-do_predict', dest='do_predict', action='store_true', default=False, help='do predicting step')
parser.add_argument('-do_ensemble', dest='do_ensemble', action='store_true', default=False, help='do predicting using ensemble')

args = parser.parse_args()
print "Overview of model training and testing settings in this run..."
print "Input size:", args.size
print "Model loss:", args.loss_param
print "Model architecture:", args.model
print "Use preprocessing:", args.pre_proc
print "Use test data augmentation:", args.test_aug
print "Use ISIC archive data to train:", args.use_archive
print "Use Ph2 data to train:", args.use_ph2
print "Do train:", args.do_train
print "Do evaluation:", args.do_evaluate
print "Do prediction:", args.do_predict
print "Do ensemble:", args.do_ensemble

size = args.size # 128, 256
pre_proc = args.pre_proc # True, False
loss_param = args.loss_param # bce, dice, jacc, mse, jacc_th
model = args.model # unet, unet2, vgg_unet, unet_b
test_aug = args.test_aug # True, False
use_archive = args.use_archive # True, False
use_ph2 = args.use_ph2 # True, False

do_train = args.do_train  # train network and save as model_name
do_evaluate = args.do_evaluate # evaluate model on val set
do_predict = args.do_predict  # use model to predict and save generated masks for Validation/Test
do_ensemble = args.do_ensemble  # use previously saved predicted masks from multiple models to generate final masks

seed = 1
height, width = size, size
nb_epoch = 220
model_name = model
batch_size = 4
monitor_metric = 'val_jacc_coef' #, 'val_jacc_coef_th'
fc_size = 4096 # for unet1,2 8192 is too large for input 256 * 256
initial_epoch = 0
metrics = [jacc_coef, jacc_coef_th]
n_channels = 3
loss = loss_options[loss_param]
optimizer = Adam(lr=1e-5)

print "Using ISIC 2018 dataset"
base_folder = "./datasets/isic2018_{}_{}".format(height, width)
image_folder = os.path.join(base_folder, "image")
mask_folder = os.path.join(base_folder, "mask")
image_names = ISIC.list_from_folder(training_folder)
if not os.path.exists(base_folder):
    ISIC.resize_images(image_names,
                       input_image_folder=training_folder,
                       input_mask_folder=training_mask_folder,
                       output_image_folder=image_folder,
                       output_mask_folder=mask_folder,
                       height=height, width=width)
train_list, val_list = train_test_split(image_names, test_size=0.1, random_state=0)

#pkl.dump(val_list, open('./datasets/val_list.pkl', 'wb'))

print "Loading images"
train, train_mask = ISIC.load_images(train_list, height, width, image_folder, mask_folder)
val, val_mask = ISIC.load_images(val_list, height, width, image_folder, mask_folder)
print "Done loading images"

if use_archive:
    print "Using ISIC Archive dataset"
    base_folder_archive = "./datasets/isic2018_archive_{}_{}".format(height, width)
    image_folder_archive = os.path.join(base_folder_archive, "image")
    mask_folder_archive = os.path.join(base_folder_archive, "mask")
    df = pd.read_csv(archive_img_seg_map)
    df = df[df['count']==1]
    imageid_list = df['image_id'].tolist()
    print "Found {} images in folder{}".format(len(imageid_list), archive_folder)
    if not os.path.exists(base_folder_archive):
        print "Creating ISIC archive dataset"
        ISIC.resize_images_archive(imageid_list,
                                   input_image_folder=archive_folder,
                                   input_mask_folder=archive_mask_folder,
                                   output_image_folder=image_folder_archive,
                                   output_mask_folder=mask_folder_archive,
                                   height=height,
                                   width=width)
    n_samples_archive = len(imageid_list)
    print "Loading images"
    filenames = [x + '.jpg' for x in imageid_list]
    train_archive, train_mask_archive = ISIC.load_images(filenames, height, width, image_folder_archive, mask_folder_archive)
    print "Done loading images"

    train = np.concatenate([train, train_archive], axis=0)
    train_mask = np.concatenate([train_mask, train_mask_archive], axis=0)

if use_ph2:
    print "Using PH2 dataset"
    base_folder_ph2 = "./datasets/isic2018_ph2_{}_{}".format(height, width)
    image_folder_ph2 = os.path.join(base_folder_ph2, "image")
    mask_folder_ph2 = os.path.join(base_folder_ph2, "mask")
    image_names = [x for x in os.listdir(ph2_folder) if x.startswith('IMD')]
    print "Found {} images in folder {}".format(len(image_names), ph2_folder)
    if not os.path.exists(base_folder_ph2):
        print "Creating PH2 dataset"
        ISIC.resize_images_ph2(image_names=image_names,
                               input_folder=ph2_folder,
                               output_image_folder=image_folder_ph2,
                               output_mask_folder=mask_folder_ph2,
                               height=height,
                               width=width)
    print "Loading images"
    filenames = [x + '.jpg' for x in image_names]
    train_ph2, train_mask_ph2 = ISIC.load_images(filenames, height, width, image_folder_ph2, mask_folder_ph2)
    print "Done loading images"
    train = np.concatenate([train, train_ph2], axis=0)
    train_mask = np.concatenate([train_mask, train_mask_ph2], axis=0)

print train.shape, train_mask.shape
print val.shape, val_mask.shape

# preprocessing using histogram equalization
if pre_proc:
    train = my_PreProc(train)
    val = my_PreProc(val)

# remove mean of training data
train_mean = np.mean(train, axis=(0, 2, 3), keepdims=True)[0]
print "train mean is", train_mean.reshape(3)
train = train - train_mean
val = val - train_mean

def model_naming(model_name, size, loss, pre_proc, use_archive, use_ph2):
    model_filename = "./weights2018/{}_{}_{}".format(model_name, size, loss_param)
    if pre_proc:
        model_filename += '_preproc'
    if use_archive:
        model_filename += '_archive'
    if use_ph2:
        model_filename += '_ph2'
    return model_filename + '.h5'

def mean_naming(model_name, size, loss, pre_proc, use_archive, use_ph2):
    mean_filename = './datasets/task1_trainmean/{}_{}_{}'.format(model_name, size, loss_param)
    if pre_proc:
        mean_filename += '_preproc'
    if use_archive:
        mean_filename += '_archive'
    if use_ph2:
        mean_filename += '_ph2'
    return mean_filename + '.pkl'

pkl.dump(train_mean, open(mean_naming(model_name, size, loss, pre_proc, use_archive, use_ph2), 'wb'))

print 'Creating model'
if model == 'unet':
    model = models.Unet(height, width, loss=loss, optimizer=optimizer, metrics=metrics, fc_size=fc_size, channels=n_channels)
elif model == 'unet2':
    model = models.Unet2(height, width, loss=loss, optimizer=optimizer, metrics=metrics, fc_size=fc_size, channels=n_channels)
elif model == 'unet_b':
    model = models.Unet_basic(height, width, loss=loss, optimizer=optimizer, metrics=metrics, channels=n_channels)
elif model == 'vgg_unet':
    model = models.VGG16_Unet(height, width, pretrained=True, freeze_pretrained=False, loss=loss, optimizer=optimizer, metrics=metrics)
else:
    print "Incorrect model name"
print "Done creating model"

vis_model = False
if vis_model:
    from keras.utils.visualize_util import plot
    print model.summary()
    plot(model, show_shapes=True, to_file='./'+model_name+'.png')

model_filename = model_naming(model_name, size, loss, pre_proc, use_archive, use_ph2)
model_checkpoint = ModelCheckpoint(model_filename, monitor=monitor_metric, save_best_only=True, verbose=1)
earlystopper = EarlyStopping(monitor=monitor_metric, patience=20, verbose=1)
print "Model will be saved in {}".format(model_filename)

def myGenerator(train_generator, train_mask_generator, show=False):
    while True:
        train_gen = next(train_generator)
        train_mask = next(train_mask_generator)

        if show: # use to show images
            for i in range(train_gen.shape[0]):
                mask = train_mask[i].reshape((width, height))
                img = train_gen[i]
                img = img[0:3]
                img = img.astype(np.uint8)
                img = img.transpose(1, 2, 0)
                f, ax = plt.subplots(1, 2)
                ax[0].imshow(img)
                ax[0].axis("off")
                ax[1].imshow(mask, cmap='Greys_r')
                ax[1].axis("off")
                plt.savefig('./generate.png')
        yield (train_gen, train_mask)

data_gen_args = dict(featurewise_center=False,
                     samplewise_center=False,
                     featurewise_std_normalization=False,
                     samplewise_std_normalization=False,
                     zca_whitening=False,
                     rotation_range=270,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     horizontal_flip=True,
                     vertical_flip=True,
                     zoom_range=0.1,
                     channel_shift_range=0,
                     fill_mode='nearest',
                     #data_format='channels_first') #keras 2
                     dim_ordering=K.image_dim_ordering()) # keras 1
print "Create Data Generator"
train_datagen = ImageDataGenerator(**data_gen_args)
train_mask_datagen = ImageDataGenerator(**data_gen_args)
train_generator = train_datagen.flow(train, batch_size=batch_size, seed=seed)
train_mask_generator = train_mask_datagen.flow(train_mask, batch_size=batch_size, seed=seed)
train_generator_f = myGenerator(train_generator, train_mask_generator)

if do_train:
    if os.path.isfile(model_filename):
        print "Model already trained in {}".format(model_filename)
        print "loading model instead"
        model.load_weights(model_filename)
    else:
        print "Training model"
        history = model.fit_generator(
            train_generator_f,
            samples_per_epoch=len(train),
            nb_epoch=nb_epoch,
            validation_data=(val, val_mask),
            callbacks=[model_checkpoint, earlystopper],
            initial_epoch=initial_epoch)
else:
    try:
        print "Load best checkpoint"
        model.load_weights(model_filename) # load best saved checkpoint
    except:
        raise Exception("Please train the model first.")

def my_predict(model, image, show=False):
    image = image + train_mean
    images = transform_img(image.transpose(1,2,0))

    if show:
        plt.figure()
        plt.imshow(image.transpose(1,2,0).astype(np.uint8))
        plt.show()
        plt.figure()
        for i in range(9):
            plt.subplot(3,3,i+1)
            plt.imshow(images[i].astype(np.uint8))
        plt.show()

    images = images.transpose(0, 3, 1, 2)
    masks = model.predict(images - train_mean)

    masks = masks.transpose(0,2,3,1)[:,:,:,0]
    masks = reverse_gt(masks)
    mask = np.mean(masks, axis=0)

    if show:
        masks = (masks * 255).astype(np.uint8)
        mask = np.mean(masks, axis=0)
        plt.figure()
        for i in range(9):
            plt.subplot(3,3,i+1)
            plt.imshow(masks[i], cmap='Greys_r')
        plt.show()
        plt.figure()
        plt.imshow(mask, cmap='Greys_r')
        plt.show()
    return mask

if do_evaluate:
    # evaluate model
    if test_aug:
        print "Using data augmentation for predicting"
        mask_pred_val = np.array([my_predict(model, val[i]) for i in trange(len(val))])
    else:
        mask_pred_val = model.predict(val)

    # save my val pred pickle
    model_name = model_filename.split('.')[1].split('/')[2]
    my_val_dir = './results/my_val_pkl'
    #if test_aug:
    #    pkl.dump(mask_pred_val, open(os.path.join(my_val_dir, model_name + '_testaug.pkl'), 'wb'))
    #else:
    #    pkl.dump(mask_pred_val, open(os.path.join(my_val_dir, model_name + '.pkl'), 'wb'))

    for pixel_threshold in [0.5]: #np.arange(0.3,1,0.05):
        mask_pred_val = np.where(mask_pred_val>=pixel_threshold, 1, 0)
        mask_pred_val = mask_pred_val * 255
        mask_pred_val = mask_pred_val.astype(np.uint8)
        dice, jacc, jacc_th = dice_jacc_mean(val_mask, mask_pred_val)
        print model_filename
        print "Resized val dice coef      : {:.4f}".format(dice)
        print "Resized val jacc coef      : {:.4f}".format(jacc)
        print "Resized val jacc_th coef   : {:.4f}".format(jacc_th)
    if test_aug:
        pkl.dump((dice, jacc, jacc_th), open(os.path.join(my_val_dir, model_name + '_testaug_score.pkl'), 'wb'))
    else:
        pkl.dump((dice, jacc, jacc_th), open(os.path.join(my_val_dir, model_name + '_score.pkl'), 'wb'))


def predict_challenge(challenge_folder, challenge_predicted_folder, plot=False):
    challenge_list = ISIC.list_from_folder(challenge_folder)
    challenge_resized_folder = challenge_folder + "_{}_{}".format(height, width)

    if not os.path.exists(challenge_resized_folder):
        print "Creating resized challenge images for prediction"
        ISIC.resize_images(challenge_list,
                           input_image_folder=challenge_folder,
                           input_mask_folder=None,
                           output_image_folder=challenge_resized_folder,
                           output_mask_folder=None,
                           height=height, width=width)

    challenge_images = ISIC.load_images(challenge_list, height, width, challenge_resized_folder)

    if pre_proc:
        challenge_images = my_PreProc(challenge_images)
    challenge_images = challenge_images - train_mean

    model_name = model_filename.split('.')[1].split('/')[2]
    try:
        if test_aug:
            mask_pred_challenge = pkl.load(open(os.path.join(challenge_predicted_folder, model_name + '_testaug.pkl'), 'rb'))
        else:
            mask_pred_challenge = pkl.load(open(os.path.join(challenge_predicted_folder, model_name + '.pkl'), 'rb'))
    except:
        model.load_weights(model_filename)
        if test_aug:
            print "Predicting using test data augmentation"
            mask_pred_challenge = np.array([my_predict(model, x) for x in tqdm(challenge_images)])
            with open(os.path.join(challenge_predicted_folder, model_name + '_testaug.pkl'), 'wb') as f:
                pkl.dump(mask_pred_challenge, f)
        else:
            print "Predicting"
            mask_pred_challenge = model.predict(challenge_images, batch_size=batch_size)
            mask_pred_challenge = mask_pred_challenge[:, 0, :, :] # remove channel dimension
            with open(os.path.join(challenge_predicted_folder, model_name + '.pkl'), 'wb') as f:
                pkl.dump(mask_pred_challenge, f)
    mask_pred_challenge = np.where(mask_pred_challenge>=0.5, 1, 0)
    mask_pred_challenge = mask_pred_challenge * 255
    mask_pred_challenge = mask_pred_challenge.astype(np.uint8)

    if not test_aug:
        challenge_predicted_folder = os.path.join(challenge_predicted_folder, model_name)
    else:
        challenge_predicted_folder = os.path.join(challenge_predicted_folder, model_name + '_testaug')
    if not os.path.exists(challenge_predicted_folder):
        os.makedirs(challenge_predicted_folder)

    print "Start predicting masks of original shapes"
    imgs = []
    mask_preds = []
    for i in trange(len(challenge_list)):
        img, mask_pred = ISIC.show_images_full_sized(challenge_list,
                                                     img_mask_pred_array=mask_pred_challenge,
                                                     image_folder=challenge_folder,
                                                     mask_folder=None,
                                                     index=i,
                                                     output_folder=challenge_predicted_folder,
                                                     plot=plot)
        #imgs.append(img)
        #mask_preds.append(mask_pred)
    return imgs, mask_preds

def join_predictions(pkl_folder, pkl_files, binary=False, threshold=0.5):
    n_pkl = float(len(pkl_files))
    array = None
    for fname in pkl_files:
        with open(os.path.join(pkl_folder,fname+".pkl"), "rb") as f:
            tmp = pkl.load(f)
            if binary:
                tmp = np.where(tmp>=threshold, 1, 0)
            if array is None:
                array = tmp
            else:
                array = array + tmp
    return array/n_pkl

if do_predict:
    # free memory
    train = None
    train_mask = None
    val = None
    test = None

    print "Start Predicting on Challenge Validation Set"
    _, _ = predict_challenge(challenge_folder=val_folder,
                             challenge_predicted_folder=val_predicted_folder,
                             plot=False)
    print "Start Predicting on Challenge Test Set"
    _, _ = predict_challenge(challenge_folder=test_folder,
                             challenge_predicted_folder=test_predicted_folder,
                             plot=False)

if do_ensemble:
    threshold = 0.5
    binary = False
    val_array = join_predictions(pkl_folder = validation_predicted_folder, pkl_files=ensemble_pkl_filenames, binary=binary, threshold=threshold)
    test_array = join_predictions(pkl_folder = test_predicted_folder, pkl_files=ensemble_pkl_filenames, binary=binary, threshold=threshold)
    model_name="ensemble_{}".format(threshold)
    for f in ensemble_pkl_filenames:
        model_name = model_name + "_" + f
    #print "Predict Validation:"
    #predict_challenge(challenge_folder=validation_folder, challenge_predicted_folder=validation_predicted_folder,
    #                    mask_pred_challenge=val_array, plot=False)
    print "Predict Test:"
    predict_challenge(challenge_folder=test_folder, challenge_predicted_folder=test_predicted_folder,
                      mask_pred_challenge=test_array, plot=False)
