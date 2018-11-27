import os
import numpy as np
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
from sklearn import metrics
from tqdm import tqdm, trange
from pre_processing import my_PreProc
from imgaug_test_cc import transform_img, reverse_gt
import argparse
import theano

theano.config.warn.round = False

np.random.seed(4)
K.set_image_dim_ordering('th')  # Theano dimension ordering: (channels, width, height)
                                # some changes will be necessary to run with tensorflow

# Folder of training images and masks
training_folder = "./datasets/ISIC2018_Task1-2_Training_Input"
training_mask_folder = "./datasets/ISIC2018_Task2_Training_GroundTruth_v3"
task1gt_folder = "./datasets/ISIC2018_Task1_Training_GroundTruth"

# Folder of validation and test images
validation_folder = "./datasets/ISIC2018_Task1-2_Validation_Input"
test_folder = "./datasets/ISIC2018_Task1-2_Test_Input"

# Folder of predicted validation and test images
task1_predict_validation_folder = './results/ISIC2018_Validation_Predicted/ensemble_128'
task1_predict_test_folder = './results/ISIC2018_Test_Predicted/ensemble_128'

loss_options = {'bce': 'binary_crossentropy',
                'dice': dice_loss,
                'jacc': jacc_loss,
                'mse': 'mean_squared_error',
                'jacc_th': jacc_loss_th}

parser = argparse.ArgumentParser(description='Process arguments for task2 step3 Unet.')
parser.add_argument('-attribute', dest='attribute', type=str, default='globules', help='attribute to model')
parser.add_argument('-size', dest='size', type=int, default=128, help='size of input image')
parser.add_argument('-loss_param', dest='loss_param', type=str, default='dice', help='loss function of model')
parser.add_argument('-model', dest='model', type=str, default='vgg_unet', help='name of model architecture')
parser.add_argument('-pre_proc', dest='pre_proc', action='store_true', default=False, help='whether using preprocessing')
parser.add_argument('-test_aug', dest='test_aug', action='store_true', default=False, help='whether using test data augmentation')

parser.add_argument('-do_train', dest='do_train', action='store_true', default=False, help='whether do training')
parser.add_argument('-do_evaluate', dest='do_evaluate', action='store_true', default=False, help='whether do evaluation')
parser.add_argument('-do_predict', dest='do_predict', action='store_true', default=False, help='whether do predicting')
parser.add_argument('-do_ensemble', dest='do_ensemble', action='store_true', default=False, help='whether do ensemble')

args = parser.parse_args()
print "Overview of model training and testing settings in this run..."
print "Attribute:", args.attribute
print 'Input size:', args.size
print 'Model loss:', args.loss_param
print 'Model architecture:', args.model
print 'Use preprocessing:', args.pre_proc
print 'Use test data augmentation:', args.test_aug
print 'Do train:', args.do_train
print 'Do evaluation:', args.do_evaluate
print 'Do prediction:', args.do_predict
print 'Do ensemble:', args.do_ensemble

attribute = args.attribute # 'pigment_network', 'negative_network', 'milia_like_cyst', 'globules', 'streaks'
size = args.size
loss_param = args.loss_param
model = args.model
test_aug = args.test_aug
pre_proc = args.pre_proc

do_train = args.do_train
do_evaluate = args.do_evaluate
do_predict = args.do_predict
do_ensemble = args.do_ensemble

# Folder to store predicted masks
validation_predicted_folder = './results_task2/ISIC2018_Validation_Predicted'
test_predicted_folder = "results_task2/ISIC2018_Test_Predicted"


seed = 1
height, width = size, size
nb_epoch = 220
model_name = model
do_filter = True # filter the segment using classifier model
batch_size = 4
monitor_metric = 'val_jacc_coef'
fc_size = 8192
initial_epoch = 0
n_channels = 3
loss = loss_options[loss_param]
optimizer = Adam(lr=1e-5)

print "Using ISIC 2018 dataset"
base_folder = "datasets/isic2018_crop_{}_{}_{}".format(attribute, height, width)
image_folder = os.path.join(base_folder, "image")
mask_folder = os.path.join(base_folder, "mask")
image_names = ISIC.list_from_folder(training_folder)
if not os.path.exists(base_folder):
    print('Cropping and Resizing...')
    ISIC.crop_resize_images(image_names,
                            input_image_folder=training_folder,
                            input_mask_folder=training_mask_folder,
                            input_gtcrop_folder=task1gt_folder,
                            output_image_folder=image_folder,
                            output_mask_folder=mask_folder,
                            height=height, width=width,
                            attribute=attribute)
train_list, val_list = train_test_split(image_names, test_size=0.1, random_state=0)

print "Loading images"
train, train_mask = ISIC.load_images(train_list, height, width, image_folder, mask_folder)
val, val_mask = ISIC.load_images(val_list, height, width, image_folder, mask_folder)
print "Done loading images"

train_label = np.sum(train_mask, axis=(1,2,3)) > 0
val_label = np.sum(val_mask, axis=(1,2,3)) > 0
print "Number of images which has attribute is {}, {}".format(np.sum(train_label), np.sum(val_label))
print "Selecting subset of training data which has %s attribute" % attribute
train = train[np.where(train_label)]
train_mask = train_mask[np.where(train_label)]
val_post = val[np.where(val_label)]
val_mask_post = val_mask[np.where(val_label)]
print train.shape, val_post.shape
print "Done loading images"

if pre_proc:
    train = my_PreProc(train)
    val = my_PreProc(val)
    val_post = my_PreProc(val_post)

train_mean = np.mean(train, axis=(0, 2, 3), keepdims=True)[0]
print "Using Train Mean: ", train_mean.reshape(3)
train = train - train_mean
val = val - train_mean
val_post = val_post - train_mean

def mean_naming2(attribute, size, pre_proc):
    mean_filename = '/home/ubuntu/isic/datasets/task2_trainmean/{}_{}'.format(attribute, size)
    if pre_proc:
        mean_filename += '_preproc'
    return mean_filename + '_crop_seg.pkl'

pkl.dump(train_mean, open(mean_naming2(attribute, size, pre_proc), 'wb'))

print 'Create model'
channels = 3
metrics = [jacc_coef]
if model == 'vgg_unet':
    model = models.VGG16_Unet(height, width, pretrained=True, freeze_pretrained=False, loss=loss, optimizer=optimizer, metrics=[jacc_coef])
elif model == 'unet':
    model = models.Unet(height, width, loss=loss, optimizer=optimizer, metrics=metrics, fc_size=4096, channels=channels)
elif model == 'unet2':
    model = models.Unet2(height, width, loss=loss, optimizer=optimizer, metrics=metrics, fc_size=4096, channels=channels)
else:
    print "Incorrect model name"

def model_naming(model_name, attribute, size, loss, pre_proc):
    model_filename = "./weights2018_task2/{}_{}_{}_{}".format(attribute, model_name, size, loss_param)
    if pre_proc:
        model_filename += '_preproc'
    return model_filename + '_crop_seg.h5'

def myGenerator(train_generator, train_mask_generator):
    while True:
        train_gen = next(train_generator)
        train_mask = next(train_mask_generator)

        if False: # use True to show images
            for i in range(train_gen.shape[0]):
                mask = train_mask[i].reshape((width, height))
                img = train_gen[i]
                img = img[0:3]
                img = img.astype(np.uint8)
                img = img.transpose(1,2,0)
                f, ax = plt.subplots(1, 2)
                ax[0].imshow(img)
                ax[0].axis("off")
                ax[1].imshow(mask, cmap='Greys_r')
                ax[1].axis("off")
                plt.savefig('./generate.png')
        yield (train_gen, train_mask)


print "Using batch size = {}".format(batch_size)
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

model_filename = model_naming(model_name, attribute, size, loss, pre_proc)
if do_train:
    if os.path.isfile(model_filename):
        print "Model already trained in", model_filename
        print "Loading model instead"
        model.load_weights(model_filename)
    else:
        print "Training model"
        model_checkpoint = ModelCheckpoint(model_filename, monitor=monitor_metric, save_best_only=True, verbose=1)
        early_stopper = EarlyStopping(monitor=monitor_metric, patience=30, verbose=1)
        history = model.fit_generator(
            train_generator_f,
            samples_per_epoch=len(train),
            nb_epoch=nb_epoch,
            validation_data=(val_post, val_mask_post),
            callbacks=[model_checkpoint, early_stopper],
            initial_epoch=initial_epoch)
else:
    print "Load best checkpoint"
    model.load_weights(model_filename) # load best saved checkpoint

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
    # save val mask groundtruth
    if size == 256:
        pkl.dump(val_mask, open('./results_task2/my_val_pkl/gt.pkl', 'wb'))

    # evaluate model
    if test_aug:
        mask_pred_val = np.array([my_predict(model, x) for x in tqdm(val)])
    else:
        mask_pred_val = model.predict(val, batch_size=batch_size)[:, 0, :, :]
    hist, bin_edges = np.histogram(mask_pred_val.flatten(), bins=5)
    print hist
    print bin_edges

    for pixel_threshold in [0.5]:#, 0.5, 0.4]:
        print "pixel_threshold is {}".format(pixel_threshold)

        mask_pred_val_256 = (np.where(mask_pred_val>=pixel_threshold, 1, 0)*255).astype(np.uint8)
        dice, jacc, jacc_th = dice_jacc_mean(val_mask, mask_pred_val_256)
        print "Resized val dice coef      : {:.4f}".format(dice)
        print "Resized val jacc coef      : {:.4f}".format(jacc)

    if do_filter:
        print 'Use Classifier model to filter the masks'
        size_1_2 = 256
        base_folder = './datasets/isic2018_crop_{}_{}_{}'.format(attribute, size_1_2, size_1_2)
        image_folder = os.path.join(base_folder, 'image')
        val = ISIC.load_images(val_list, size_1_2, size_1_2, image_folder)
        # filter by pre_cls model
        pre_cls_model_filename = './weights2018_task2/{}_{}_crop_pre_cls.h5'.format('vgg', size_1_2)
        pre_cls_model = models.VGG16(size_1_2, size_1_2, pretrained=True, freeze_pretrained=False, loss='binary_crossentropy', optimizer=Adam(lr=1e-5), metrics=['accuracy'])
        pre_cls_model.load_weights(pre_cls_model_filename)
        step1_mean = pkl.load(open('./datasets/task2_step1_train_mean.pkl', 'rb'))
        prob_pre_val = pre_cls_model.predict(val - step1_mean)
        print prob_pre_val.shape
        # filter by cls model
        cls_model_filename = './weights2018_task2/{}_{}_{}_crop_cls.h5'.format(attribute, 'vgg', size_1_2)
        cls_model = models.VGG16(size_1_2, size_1_2, pretrained=True, freeze_pretrained=False, loss='binary_crossentropy', optimizer=Adam(lr=1e-5), metrics=['accuracy'])
        cls_model.load_weights(cls_model_filename)
        step2_mean = pkl.load(open('./datasets/task2_step2_train_mean.pkl', 'rb'))
        prob_pred_val = cls_model.predict(val - step2_mean)
        print prob_pred_val.shape

        mask_pred_val = mask_pred_val * prob_pred_val[:, np.newaxis] * prob_pre_val[:, np.newaxis]

        for pixel_threshold in [0.5]:#, 0.5, 0.4]:
            print "pixel_threshold is {}".format(pixel_threshold)
            mask_pred_val_256 = (np.where(mask_pred_val>=pixel_threshold, 1, 0)*255).astype(np.uint8)
            dice, jacc, jacc_th = dice_jacc_mean(val_mask, mask_pred_val_256)
            print "Resized val dice coef      : {:.4f}".format(dice)
            print "Resized val jacc coef      : {:.4f}".format(jacc)

        model_name = model_filename.split('.')[1].split('/')[2]
        if test_aug:
            with open(os.path.join('./results_task2/my_val_pkl', model_name + '_testaug.pkl'), 'wb') as f:
                pkl.dump(mask_pred_val, f)
        else:
            with open(os.path.join('./results_task2/my_val_pkl', model_name + '.pkl'), 'wb') as f:
                pkl.dump(mask_pred_val, f)

def predict_challenge(challenge_folder, challenge_predicted_folder, task1predicted_folder, plot=False):
    challenge_list = ISIC.list_from_folder(challenge_folder)
    challenge_crop_resized_folder = challenge_folder + "_crop_{}_{}".format(height, width)
    crop_inds_file = os.path.join(challenge_crop_resized_folder, 'inds.pkl')
    if not os.path.exists(challenge_crop_resized_folder) or not os.path.isfile(crop_inds_file):
        print "Cropping and resizing images"
        inds = ISIC.crop_resize_images(challenge_list,
                                       input_image_folder=challenge_folder,
                                       input_mask_folder=None,
                                       input_gtcrop_folder=task1predicted_folder,
                                       output_image_folder=challenge_crop_resized_folder,
                                       output_mask_folder=None,
                                       height=height, width=width)
        pkl.dump(inds, open(crop_inds_file, 'wb'))
    inds = pkl.load(open(crop_inds_file, 'rb'))

    challenge_images = ISIC.load_images(challenge_list, height, width, challenge_crop_resized_folder)

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
        print 'making prediction...'
        model.load_weights(model_filename)
        if test_aug:
            print "Predicting using test data augmentation"
            mask_pred_challenge = np.array([my_predict(model, x) for x in tqdm(challenge_images)])
            print mask_pred_challenge.shape
        else:
            mask_pred_challenge = model.predict(challenge_images, batch_size=batch_size)
            mask_pred_challenge = mask_pred_challenge[:, 0, :, :]
            print mask_pred_challenge.shape
        if do_filter:
            print "Using step2 and step1 classifiers"
            size_1_2 = 256
            challenge_crop_resized_folder = challenge_folder + "_crop_{}_{}".format(size_1_2, size_1_2)
            if not os.path.exists(challenge_crop_resized_folder):
                print "Cropping and resizing images"
                _ = ISIC.crop_resize_images(challenge_list,
                                            input_image_folder=challenge_folder,
                                            input_mask_folder=None,
                                            input_gtcrop_folder=task1predicted_folder,
                                            output_image_folder=challenge_crop_resized_folder,
                                            output_mask_folder=None,
                                            height=size_1_2, width=size_1_2)

            challenge_images = ISIC.load_images(challenge_list, size_1_2, size_1_2, challenge_crop_resized_folder)
            # filter by pre model
            pre_cls_model_filename = './weights2018_task2/{}_{}_crop_pre_cls.h5'.format('vgg', size_1_2)
            pre_cls_model = models.VGG16(size_1_2, size_1_2, pretrained=True, freeze_pretrained=False, loss='binary_crossentropy', optimizer=Adam(lr=1e-5), metrics=['accuracy'])
            pre_cls_model.load_weights(pre_cls_model_filename)
            step1_mean = pkl.load(open('./datasets/task2_step1_train_mean.pkl', 'rb'))
            prob_pre = pre_cls_model.predict(challenge_images - step1_mean)
            print prob_pre.shape
            # filter by cls model
            cls_model_filename = './weights2018_task2/{}_{}_{}_crop_cls.h5'.format(attribute, 'vgg', size_1_2)
            cls_model = models.VGG16(size_1_2, size_1_2, pretrained=True, freeze_pretrained=False, loss='binary_crossentropy', optimizer=Adam(lr=1e-5), metrics=['accuracy'])
            cls_model.load_weights(cls_model_filename)
            step2_mean = pkl.load(open('./datasets/task2_step2_train_mean.pkl', 'rb'))
            prob_pred = cls_model.predict(challenge_images - step2_mean)
            print prob_pred.shape
            # prob product
            mask_pred_challenge = mask_pred_challenge * prob_pred[:, np.newaxis] * prob_pre[:, np.newaxis]
        if test_aug:
            with open(os.path.join(challenge_predicted_folder, model_name + '_testaug.pkl'), 'wb') as f:
                pkl.dump(mask_pred_challenge, f)
        else:
            with open(os.path.join(challenge_predicted_folder, model_name + '.pkl'), 'wb') as f:
                pkl.dump(mask_pred_challenge, f)

    cutoff = 0.5
    mask_pred_challenge_256 = (np.where(mask_pred_challenge>=cutoff, 1, 0) * 255).astype(np.uint8)

    if test_aug:
        challenge_predicted_folder = os.path.join(challenge_predicted_folder, model_name + '_testaug')
    else:
        challenge_predicted_folder = os.path.join(challenge_predicted_folder, model_name)
    if not os.path.exists(challenge_predicted_folder):
        os.makedirs(challenge_predicted_folder)

    print "Start predicting masks of original shapes"
    imgs = []
    mask_preds = []
    for i in trange(len(challenge_list)):
        img, mask_pred = ISIC.show_crop_images_full_sized(challenge_list,
                                                          img_mask_pred_array=mask_pred_challenge_256,
                                                          image_folder=challenge_folder,
                                                          inds=inds,
                                                          mask_folder=None,
                                                          index=i,
                                                          output_folder=challenge_predicted_folder,
                                                          attribute=attribute,
                                                          plot=plot)

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

    print "Start Challenge Validation"
    predict_challenge(challenge_folder=validation_folder,
                      challenge_predicted_folder=validation_predicted_folder,
                      task1predicted_folder=task1_predict_validation_folder,
                      plot=False)

    print "Start Challenge Test"
    predict_challenge(challenge_folder=test_folder,
                      challenge_predicted_folder=test_predicted_folder,
                      task1predicted_folder=task1_predict_test_folder,
                      plot=False)


