import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import shutil
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm, trange

np.random.seed(4)


def get_mask(image_name, mask_folder, attribute=None, rescale_mask=True):
    # for original masks
    if not attribute:
        img_mask = cv2.imread(os.path.join(mask_folder, image_name.replace(".jpg", "_segmentation.png")), cv2.IMREAD_GRAYSCALE)
    else:
        img_mask = cv2.imread(os.path.join(mask_folder, image_name.replace(".jpg", "_attribute_%s.png" % attribute)), cv2.IMREAD_GRAYSCALE)
    # for resized masks renamed in png
    if img_mask is None:
        img_mask = cv2.imread(os.path.join(mask_folder, image_name.replace(".jpg", ".png")), cv2.IMREAD_GRAYSCALE)
    # threshold mask
    _, img_mask = cv2.threshold(img_mask, 127, 255, cv2.THRESH_BINARY)
    if rescale_mask:
        img_mask = img_mask / 255.
    return img_mask

def get_color_image(image_name, image_folder):
    img = cv2.imread(os.path.join(image_folder, image_name.replace(".jpg", ".png")))
    if img is None:
        img = cv2.imread(os.path.join(image_folder, image_name))
    # BGR to GRB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    # channel last to channel first
    img = img.transpose((2,0,1)).astype(np.float32)
    return img

def load_images(images_list, height, width, image_folder, mask_folder=None):
    n_chan = 3
    img_array = np.zeros((len(images_list), n_chan, height, width), dtype=np.float32)
    if mask_folder:
        img_mask_array = np.zeros((len(images_list), height, width), dtype=np.float32)
    for i, image_name in enumerate(images_list):
        img = get_color_image(image_name, image_folder)
        img_array[i] = img
        if mask_folder:
            img_mask = get_mask(image_name, mask_folder)
            img_mask_array[i] =img_mask
    if mask_folder:
        return (img_array, img_mask_array.astype(np.uint8)[:, np.newaxis, :, :])
    else:
        return img_array

def list_from_folder(image_folder):
    image_list = []
    for image_filename in os.listdir(image_folder):
        if image_filename.endswith(".jpg"):
            image_list.append(image_filename)
    print "Found {} images in folder {}.".format(len(image_list), image_folder)
    return image_list

def move_images(images_list, input_image_folder, input_mask_folder, output_image_folder, output_mask_folder, height=None, width=None, attribute=None, same_name=False):
    if not os.path.exists(output_image_folder):
        os.makedirs(output_image_folder)
    if input_mask_folder and not os.path.exists(output_mask_folder):
        os.makedirs(output_mask_folder)
    for k in tqdm(range(len(images_list))):
        image_filename = images_list[k]
        image_name = os.path.basename(image_filename).split('.')[0]
        if height and width:
            img = cv2.imread(os.path.join(input_image_folder, image_filename))
            img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(os.path.join(output_image_folder, image_name + ".png"), img)
            if input_mask_folder:
                img_mask = get_mask(image_filename, input_mask_folder, attribute=attribute, rescale_mask=False)
                img_mask = cv2.resize(img_mask, (width, height), interpolation = cv2.INTER_CUBIC)
                _, img_mask = cv2.threshold(img_mask, 127, 255, cv2.THRESH_BINARY)
                cv2.imwrite(os.path.join(output_mask_folder, image_name + ".png"), img_mask)
        else:
            if not same_name:
                shutil.copyfile(os.path.join(input_image_folder, image_filename), os.path.join(output_image_folder,image_name+".jpg"))
            else:
                img = cv2.imread(os.path.join(input_image_folder,image_filename))
                cv2.imwrite(os.path.join(output_image_folder,image_name+".png"), img)

            if input_mask_folder:
                if not attribute:
                    image_mask_filename = image_filename.replace(".jpg","_segmentation.png")
                else:
                    image_mask_filename = image_filename.replace(".jpg", "_attribute_%s.png" % attribute)
                shutil.copyfile(os.path.join(input_mask_folder,image_mask_filename), os.path.join(output_mask_folder,image_name+".png"))

def resize_images(images_list, input_image_folder, input_mask_folder, output_image_folder, output_mask_folder, height, width, attribute=None):
    return move_images(images_list, input_image_folder, input_mask_folder, output_image_folder, output_mask_folder, height, width, attribute)

def resize_images_archive(images_list, input_image_folder, input_mask_folder, output_image_folder, output_mask_folder, height, width):
    if not os.path.exists(output_image_folder):
        os.makedirs(output_image_folder)
    if input_mask_folder and not os.path.exists(output_mask_folder):
        os.makedirs(output_mask_folder)
    img_seg_map = pd.read_csv('./datasets/Archive/isic_archive_img_seg/img_seg_count.csv')
    for k in tqdm(range(len(images_list))):
        image_name = images_list[k]
        image_filename = image_name + '.jpg'
        img = cv2.imread(os.path.join(input_image_folder,image_filename))
        img = cv2.resize(img, (width, height), interpolation = cv2.INTER_CUBIC)
        cv2.imwrite(os.path.join(output_image_folder,image_name+".png"), img)
        if input_mask_folder:
            image_seg_id = img_seg_map[img_seg_map['image_id']==image_name].iloc[0]['seg_id']
            image_seg_filename = image_name + '_' + image_seg_id + '.png'
            img_mask = cv2.imread(os.path.join(input_mask_folder, image_seg_filename), cv2.IMREAD_GRAYSCALE)
            _, img_mask = cv2.threshold(img_mask,127,255,cv2.THRESH_BINARY)
            img_mask = cv2.resize(img_mask, (width, height), interpolation = cv2.INTER_CUBIC)
            _, img_mask = cv2.threshold(img_mask,127,255,cv2.THRESH_BINARY)
            cv2.imwrite(os.path.join(output_mask_folder,image_name+".png"), img_mask)

def resize_images_ph2(image_names, input_folder, output_image_folder, output_mask_folder, height=None, width=None, same_name=False):
    N = len(image_names)
    if not os.path.exists(output_image_folder):
        os.makedirs(output_image_folder)
    if not os.path.exists(output_mask_folder):
        os.makedirs(output_mask_folder)
    for k in tqdm(range(N)):
        image_name = image_names[k]
        # image
        img = cv2.imread(os.path.join(input_folder, image_name, image_name + "_Dermoscopic_Image", image_name + ".bmp"))
        img = cv2.resize(img, (width, height), interpolation = cv2.INTER_CUBIC)
        cv2.imwrite(os.path.join(output_image_folder, image_name+".png"), img)
        # mask
        img_mask = cv2.imread(os.path.join(input_folder, image_name, image_name + "_lesion", image_name + "_lesion.bmp"), cv2.IMREAD_GRAYSCALE)
        _, img_mask = cv2.threshold(img_mask, 127, 255, cv2.THRESH_BINARY)
        img_mask = cv2.resize(img_mask, (width, height), interpolation = cv2.INTER_CUBIC)
        _, img_mask = cv2.threshold(img_mask, 127, 255, cv2.THRESH_BINARY)
        cv2.imwrite(os.path.join(output_mask_folder, image_name+".png"), img_mask)
    return image_names

def crop_resize_images(images_list, input_image_folder, input_mask_folder, input_gtcrop_folder, output_image_folder, output_mask_folder, height, width, attribute=None):
    if not os.path.exists(output_image_folder):
        os.makedirs(output_image_folder)
    if input_mask_folder and not os.path.exists(output_mask_folder):
        os.makedirs(output_mask_folder)
    inds = []
    for k in trange(len(images_list)):
        image_filename = images_list[k]
        image_name = os.path.basename(image_filename).split('.')[0]
        img = cv2.imread(os.path.join(input_image_folder, image_filename))
        gt_mask = get_mask(image_filename, input_gtcrop_folder, rescale_mask=False)
        if input_mask_folder:
            img_mask = get_mask(image_filename, input_mask_folder, attribute=attribute, rescale_mask=False)

        # cropping by gt_mask
        gt_mask_colsum = np.sum(gt_mask, axis=0).astype(bool)
        col_index = np.where(gt_mask_colsum>0)
        gt_mask_rowsum = np.sum(gt_mask, axis=1).astype(bool)
        row_index = np.where(gt_mask_rowsum>0)

        ind = [row_index[0][0], row_index[0][-1]+1, col_index[0][0], col_index[0][-1]+1]
        inds.append(ind)
        img = img[ind[0]:ind[1], ind[2]:ind[3]]
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(os.path.join(output_image_folder, image_name + ".png"), img)
        if input_mask_folder and output_mask_folder:
            img_mask = img_mask[ind[0]:ind[1], ind[2]:ind[3]]
            img_mask = cv2.resize(img_mask, (width, height), interpolation=cv2.INTER_CUBIC)
            _, img_mask = cv2.threshold(img_mask, 127, 255, cv2.THRESH_BINARY)
            cv2.imwrite(os.path.join(output_mask_folder, image_name + ".png"), img_mask)

    return inds


def get_mask_full_sized(mask_pred, original_shape, output_folder=None, image_name=None):
    mask_pred = cv2.resize(mask_pred, (original_shape[2], original_shape[1])) # resize to original mask size
    _, mask_pred = cv2.threshold(mask_pred, 127, 255, cv2.THRESH_BINARY)
    if output_folder and image_name:
        cv2.imwrite(os.path.join(output_folder, image_name.split('.')[0]+"_segmentation.png"), mask_pred)
    return mask_pred

def show_images_full_sized(image_list, img_mask_pred_array, image_folder, mask_folder, index, output_folder=None, plot=True):
    image_name = image_list[index]
    img = get_color_image(image_name, image_folder).astype(np.uint8)
    if mask_folder:
        mask_true = get_mask(image_name, mask_folder, rescale_mask=False)
    mask_pred = get_mask_full_sized(img_mask_pred_array[index],
                                    img.shape,
                                    output_folder=output_folder,
                                    image_name=image_name)
    if mask_folder:
        if plot:
            f, ax = plt.subplots(1, 3)
            img = img.transpose(1, 2, 0)
            ax[0].imshow(img); ax[0].axis("off");
            ax[1].imshow(mask_true, cmap='Greys_r');  ax[1].axis("off");
            ax[2].imshow(mask_pred, cmap='Greys_r'); ax[2].axis("off"); plt.show()
        return img, mask_true, mask_pred
    else:
        if plot:
            f, ax = plt.subplots(1, 2)
            img = img.transpose(1, 2, 0)
            ax[0].imshow(img); ax[0].axis("off");
            ax[1].imshow(mask_pred, cmap='Greys_r'); ax[1].axis("off"); plt.show()
        return img, mask_pred

def get_crop_mask_full_sized(mask_pred, original_shape, ind, output_folder, image_name, attribute):
    full_mask_pred = np.zeros((original_shape[1], original_shape[2]))
    crop_mask_pred = cv2.resize(mask_pred, (ind[3]-ind[2], ind[1]-ind[0])) # resize to original mask size
    full_mask_pred[ind[0]:ind[1], ind[2]:ind[3]] = crop_mask_pred
    _, mask_pred = cv2.threshold(full_mask_pred, 127, 255, cv2.THRESH_BINARY)
    if output_folder and image_name:
        cv2.imwrite(os.path.join(output_folder, image_name.split('.')[0]+"_attribute_{}.png".format(attribute)), mask_pred)
    return mask_pred


def show_crop_images_full_sized(image_list, img_mask_pred_array, image_folder, inds, mask_folder, index, output_folder, attribute, plot=True):
    image_name = image_list[index]
    ind = inds[index]
    img = get_color_image(image_name, image_folder).astype(np.uint8)
    if mask_folder:
        mask_true = get_mask(image_name, mask_folder, rescale_mask=False)
    mask_pred = get_crop_mask_full_sized(img_mask_pred_array[index],
                                         img.shape,
                                         ind,
                                         output_folder=output_folder,
                                         image_name=image_name,
                                         attribute=attribute)
    if mask_folder:
        if plot:
            f, ax = plt.subplots(1, 3)
            img = img.transpose(1, 2, 0)
            ax[0].imshow(img); ax[0].axis("off");
            ax[1].imshow(mask_true, cmap='Greys_r');  ax[1].axis("off");
            ax[2].imshow(mask_pred, cmap='Greys_r'); ax[2].axis("off"); plt.show()
        return img, mask_true, mask_pred
    else:
        if plot:
            f, ax = plt.subplots(1, 2)
            img = img.transpose(1, 2, 0)
            ax[0].imshow(img); ax[0].axis("off");
            ax[1].imshow(mask_pred, cmap='Greys_r'); ax[1].axis("off"); plt.show()
        return img, mask_pred


def img_mask_pred(image_list, img_mask_pred_array, image_folder, mask_folder, index, output_folder):
    image_name = image_list[index]
    img = get_color_image(image_name, image_folder, remove_mean_imagenet=False).astype(np.uint8)
    img = img.transpose(1,2,0)
    mask_true = get_mask(image_name, mask_folder, rescale_mask=False)
    mask_pred = get_mask(image_name.split('.')[0]+"_segmentation.png", output_folder, rescale_mask=False)
    print np.max(mask_true), mask_true.shape
    print np.max(mask_pred), mask_pred.shape
    return img, mask_true, mask_pred


