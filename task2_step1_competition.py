import os
import numpy as np
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import pickle as pkl
import ISIC_dataset as ISIC
from metrics import dice_loss, jacc_loss, jacc_coef, dice_jacc_mean
import models
from sklearn.model_selection import train_test_split
from sklearn import metrics

np.random.seed(4)
K.set_image_dim_ordering('th')  # Theano dimension ordering: (channels, width, height)
                                # some changes will be necessary to run with tensorflow

# Folder of training images and masks
training_folder = "./datasets/ISIC2018_Task1-2_Training_Input"
training_mask_folder = "./datasets/ISIC2018_Task2_Training_GroundTruth_v3"
task1gt_folder = './datasets/ISIC2018_Task1_Training_GroundTruth'

# Folder of validation and test images
validation_folder = "./datasets/ISIC2018_Task1-2_Validation_Input"
test_folder = "./datasets/ISIC2018_Task1-2_Test_Input"

seed = 1
size = 128
height, width = size, size
nb_epoch = 220
model = 'vgg'
model_name = model

do_train = True
do_evaluate = True
batch_size = 16
monitor_metric = 'val_loss'
initial_epoch = 0
n_channels = 3

print "Using ISIC 2018 dataset"
image_names = ISIC.list_from_folder(training_folder)
train_list, val_list = train_test_split(image_names, test_size=0.1, random_state=0)
train_labels, val_labels = [], []
for attribute in ['globules', 'milia_like_cyst', 'negative_network', 'pigment_network', 'streaks']:
    print "Processing {}".format(attribute)
    base_folder = "datasets/isic2018_crop_{}_{}_{}".format(attribute, height, width)
    image_folder = os.path.join(base_folder, "image")
    mask_folder = os.path.join(base_folder, "mask")
    if not os.path.exists(base_folder):
        print 'Cropping and Resizing...'
        inds = ISIC.crop_resize_images(image_names,
                                       input_image_folder=training_folder,
                                       input_mask_folder=training_mask_folder,
                                       input_gtcrop_folder=task1gt_folder,
                                       output_image_folder=image_folder,
                                       output_mask_folder=mask_folder,
                                       height=height, width=width,
                                       attribute=attribute)
        print "Saving cropped index..."
        pkl.dump(inds, open('./datasets/isic2018_crop_ind_train.pkl', 'wb'))

    print "Loading images"
    train, train_mask = ISIC.load_images(train_list, height, width, image_folder, mask_folder)
    val, val_mask = ISIC.load_images(val_list, height, width, image_folder, mask_folder)
    print (train.shape, val.shape)
    train_label = np.sum(train_mask, axis=(1, 2, 3)) > 0
    val_label = np.sum(val_mask, axis=(1, 2, 3)) > 0
    print (np.sum(train_label), np.sum(val_label))
    print "Done loading {} images".format(attribute)
    train_labels.append(train_label)
    val_labels.append(val_label)

print np.array(train_labels).shape
train_label = np.sum(np.array(train_labels), axis=0) > 0
val_label = np.sum(np.array(val_labels), axis=0) > 0
print (np.sum(train_label), np.sum(val_label))
print 'Saving pre-classify labels'
pkl.dump((train_label, val_label), open('./datasets/crop_preclasslabels.pkl', 'wb'))


train_mean = np.mean(train, axis=(0, 2, 3), keepdims=True)[0]
print "Using Train Mean:", train_mean.reshape(3)
train = train - train_mean
val = val - train_mean
print "Saving task2 step1 train mean"
pkl.dump(train_mean, open('./datasets/task2_step1_train_mean.pkl', 'wb'))

optimizer = Adam(lr=1e-5)
model_filename = "weights2018_task2/{}_{}_crop_pre_cls.h5".format(model_name, size)

print 'Create model'
if model == 'vgg':
    loss = 'binary_crossentropy'
    model = models.VGG16(height, width, pretrained=True, freeze_pretrained=False, loss=loss, optimizer=optimizer, metrics=['accuracy'])
else:
    print "Incorrect model name"

vis_model = False
if vis_model:
    from keras.utils.visualize_util import plot
    print model.summary()
    plot(model, show_shapes=True, to_file='./'+model_name+'.png')
print haha

def myGenerator(train_generator, show=False):
    while True:
        train_gen = next(train_generator)

        if show: # use True to show images
            for i in range(train_gen.shape[0]):
                img = train_gen[i]
                img = img[0:3]
                img = img.astype(np.uint8)
                img = img.transpose(1, 2, 0)
                f, ax = plt.subplots(1, 2)
                ax[0].imshow(img)
                ax[0].axis("off")
                plt.savefig('./generate.png')
        yield train_gen


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
train_generator = train_datagen.flow(train, train_label, batch_size=batch_size, seed=seed)
train_generator_f = myGenerator(train_generator)

if do_train:
    print "Training model"
    model_checkpoint = ModelCheckpoint(model_filename, monitor=monitor_metric, save_best_only=True, verbose=1)
    early_stopper = EarlyStopping(monitor=monitor_metric, patience=30, verbose=1)
    history = model.fit_generator(train_generator_f,
                                  samples_per_epoch=len(train),
                                  nb_epoch=nb_epoch,
                                  validation_data=(val, val_label),
                                  callbacks=[model_checkpoint, early_stopper],
                                  initial_epoch=initial_epoch)
else:
    print "Load best checkpoint"
    model.load_weights(model_filename) # load best saved checkpoint

if do_evaluate:
    # evaluate model
    prob_pred_train = model.predict(train)
    prob_pred_val = model.predict(val)
    print np.mean(prob_pred_train)
    print np.mean(prob_pred_val)

    auc = metrics.roc_auc_score(train_label, prob_pred_train)
    print "AUROC train     : {:.4f}".format(auc)
    prc = metrics.average_precision_score(train_label, prob_pred_train)
    print "AUPRC train     : {:.4f}".format(prc)

    auc = metrics.roc_auc_score(val_label, prob_pred_val)
    print "AUROC val       : {:.4f}".format(auc)
    prc = metrics.average_precision_score(val_label, prob_pred_val)
    print "AUPRC val       : {:.4f}".format(prc)

    fpr, tpr, thresholds = metrics.roc_curve(train_label, prob_pred_train)
    opt = np.argmin((fpr-0)**2 + (tpr-1)**2)
    cutoff_train = thresholds[opt]
    print "Best cutoff in train set is {}".format(cutoff_train)

    fpr, tpr, thresholds = metrics.roc_curve(val_label, prob_pred_val)
    opt = np.argmin((fpr-0)**2 + (tpr-1)**2)
    cutoff_val = thresholds[opt]
    print "Best cutoff in val set is {}".format(cutoff_val)

    for cutoff in [cutoff_train, cutoff_val]:
        print 'cutoff: ', cutoff

        label_pred_train = (prob_pred_train>cutoff).astype(int)
        confusion_matrix = metrics.confusion_matrix(train_label, label_pred_train)
        acc = metrics.accuracy_score(train_label, label_pred_train)
        print confusion_matrix
        print "Accuracy train   : {:.4f}".format(acc)

        label_pred_val = (prob_pred_val>cutoff).astype(int)
        confusion_matrix = metrics.confusion_matrix(val_label, label_pred_val)
        acc = metrics.accuracy_score(val_label, label_pred_val)
        print confusion_matrix
        print "Accuracy val     : {:.4f}".format(acc)
