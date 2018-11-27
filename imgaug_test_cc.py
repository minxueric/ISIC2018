import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage import transform
import copy


def aug_img(image, image_groundtruth):
    '''
    :param image: input image
    :param image_groudtruth: ground truth of input image
    :return:
    16 augmented images of shape(16, height, width, 3) and its ground truth images of shape(16, height, width)
    '''
    ia.seed(1)
    # dst = transform.rescale(image, 0.2)
    images = np.array(
        [image for _ in range(16)]
    )

    # dst_groundtruth = transform.rescale(image_groudtruth, 0.2)
    images_groundtruth = np.array(
        [image_groundtruth for _ in range(16)]
    )

    seq = iaa.Sequential([
        iaa.Fliplr(0.5), # horizontal flips
        iaa.Flipud(0.5),
        iaa.Crop(percent=(0, 0.01)), # random crops
        # Small gaussian blur with random sigma between 0 and 0.5.
        # But we only blur about 50% of all images.
        iaa.Sometimes(0.5,
            iaa.GaussianBlur(sigma=(0, 0.5))
        ),
        # Make some images brighter and some darker.
        # In 20% of all cases, we sample the multiplier once per channel,
        # which can end up changing the color of the images.
        iaa.Multiply((0.8, 1.2), per_channel=0.2),
        # Apply affine transformations to each image.
        # Scale/zoom them, translate/move them, rotate them and shear them.
        iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            # translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            rotate=(-25, 25),
            shear=(-8, 8)
        )
    ], random_order=True) # apply augmenters in random order

    seq_det = seq.to_deterministic()
    images = seq_det.augment_images(images)
    images_groundtruth = seq_det.augment_images(images_groundtruth)
    images_groundtruth[images_groundtruth > 0] = 1

    return images, images_groundtruth

iaas = [[iaa.Noop(), iaa.Noop()],
        [iaa.Affine(rotate=30), iaa.Affine(rotate=-30)],
        [iaa.Affine(rotate=60), iaa.Affine(rotate=-60)],
        [iaa.Affine(translate_px={"x": 10}), iaa.Affine(translate_px={"x": -10})],
        [iaa.Affine(translate_px={"x": -10}), iaa.Affine(translate_px={"x": 10})],
        [iaa.Affine(translate_px={"y": 10}), iaa.Affine(translate_px={"y": -10})],
        [iaa.Affine(translate_px={"y": -10}), iaa.Affine(translate_px={"y": 10})],
        [iaa.Fliplr(1.0), iaa.Fliplr(1.0)],
        [iaa.Flipud(1.0), iaa.Flipud(1.0)]]


def transform_img(image, iaas=iaas):
    '''
    :param image: input image
    :param image_groudtruth: ground truth of input image
    :return:
    '''
    ia.seed(1)
    images = np.array([image for _ in range(len(iaas))])

    for i in range(len(iaas)):
        seq = iaa.Sequential([iaas[i][0]], random_order=True) # apply augmenters in random order
        seq_det = seq.to_deterministic()
        images[i:i+1] = seq_det.augment_images(images[i:i+1])

    return images

def reverse_gt(images_groundtruth):
    '''
    :param image: input image
    :param image_groudtruth: ground truth of input image
    :return:
    '''
    ia.seed(1)
    images_groundtruth_reverse = copy.copy(images_groundtruth)

    for i in range(len(iaas)):
        seq_reverse = iaa.Sequential([iaas[i][1]], random_order=True)
        seq_reverse_det = seq_reverse.to_deterministic()

        images_groundtruth_reverse[i:i+1] = seq_reverse_det.augment_images(images_groundtruth[i:i+1])

    return images_groundtruth_reverse


if __name__ == '__main__':
    #image = io.imread('/Users/chechao/PycharmProjects/ISIC/ISIC-2018/ISIC2018_Task1-2_Training_Input/ISIC_0000008.jpg')
    image = io.imread('./datasets/ISIC2018_Task1-2_Training_Input/ISIC_0000000.jpg')
    print (image.shape)
    image_groundtruth = io.imread('./datasets/ISIC2018_Task1_Training_GroundTruth/ISIC_0000000_segmentation.png')

    images, images_groundtruth, images_reverse, images_groundtruth_reverse = aug_img_reverse(image, image_groundtruth)

    print images.shape, images_reverse.shape

    images = np.concatenate([image[np.newaxis,:,:,:], images, images_reverse], axis=0)
    images_groundtruth = np.concatenate([image_groundtruth[np.newaxis,:,:], images_groundtruth, images_groundtruth_reverse], axis=0)

    for i in range(1+4*2):
        plt.subplot(9, 2, 2 * i + 1)
        plt.imshow(images[i])
        plt.title(str(i))
        plt.axis('off')
        plt.subplot(9, 2, 2 * i + 2)
        plt.imshow(images_groundtruth[i])
        plt.title(str(i))
        plt.axis('off')
    plt.show()
    print(images.shape)
    print(images_groundtruth.shape)
    # print(np.unique(images_groundtruth))
