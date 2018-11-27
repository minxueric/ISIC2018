import numpy as np
from keras import backend as K
from sklearn.metrics import jaccard_similarity_score

smooth_default = 1.

def dice_coef(y_true, y_pred, smooth=smooth_default, per_batch=True):
    if not per_batch:
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    else:
        y_true_f = K.batch_flatten(y_true)
        y_pred_f = K.batch_flatten(y_pred)
        intersec = 2. * K.sum(y_true_f * y_pred_f, axis=1, keepdims=True) + smooth
        union = K.sum(y_true_f, axis=1, keepdims=True) + K.sum(y_pred_f, axis=1, keepdims=True) + smooth
        return K.mean(intersec / union)

def jacc_coef(y_true, y_pred, smooth=smooth_default):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + smooth)

def jacc_coef_th(y_true, y_pred, smooth=smooth_default):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    jacc = (intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + smooth)
    result = K.switch(
        jacc > 0.65,
        jacc,
        jacc * 0.1
    )
    return result

def jacc_loss(y_true, y_pred):
    return -jacc_coef(y_true, y_pred)

def jacc_loss_th(y_true, y_pred):
    return -jacc_coef_th(y_true, y_pred)

def dice_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def dice_jacc_single(mask_true, mask_pred, smooth=smooth_default):
    bool_true = mask_true.reshape(-1).astype(np.bool)
    bool_pred = mask_pred.reshape(-1).astype(np.bool)
    if bool_true.shape != bool_pred.shape:
        raise ValueError("Masks of different sizes.")

    bool_sum = bool_true.sum() + bool_pred.sum()
    # if bool_sum == 0:
        # print "Empty mask"
        # return 0,0
    intersec = np.logical_and(bool_true, bool_pred).sum()
    dice = (2. * intersec + smooth) / (bool_sum + smooth)
    # jacc = jaccard_similarity_score(bool_true.reshape((1, -1)), bool_pred.reshape((1, -1)), normalize=True, sample_weight=None)
    jacc = float(intersec + smooth) / (bool_sum - intersec + smooth)
    return dice, jacc

def dice_jacc_mean(mask_true, mask_pred, smooth=smooth_default):
    dice = 0
    jacc = 0
    jacc_th = 0
    n = len(mask_true)
    for i in range(n):
        current_dice, current_jacc = dice_jacc_single(mask_true=mask_true[i], mask_pred=mask_pred[i], smooth=smooth)
        # threshold Jaccard index metric
        jacc += current_jacc
        current_jacc = 0  if current_jacc < 0.65 else current_jacc
        dice += current_dice
        jacc_th += current_jacc
    return dice/n, jacc/n, jacc_th/n
