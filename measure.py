from keras import backend as K
import numpy as np


def f_precision(y_mask, y_pred):
    true_pos = np.sum(np.clip(y_mask * y_pred, 0, 1))
    #print("true_pos=", true_pos)
    pred_pos = np.sum(y_pred)
    #print("pred_pos=", pred_pos)

    precision = true_pos / pred_pos
    return precision

def f_recall(y_mask, y_pred):
    true_pos = np.sum(np.clip(y_mask * y_pred, 0, 1))
    mask_pos = np.sum(y_mask)
    #print("mask_pos=", mask_pos)

    recall = true_pos / mask_pos
    return recall


#y_mask = a list of 0 or 1, where 1 denotes the pixel of region of interest
#y_pred = the same as a

def f_f1_score(y_mask, y_pred):
    precision = f_precision(y_mask, y_pred)
    recall = f_recall(y_mask, y_pred)
    return 2*((precision*recall)/(precision+recall))

'''
y_mask = np.array([[1,0], [0,1]])
y_pred = np.array([[1,0], [1,0]])

f1_s = f_f1_score(y_mask, y_pred)
print("f1_s=", f1_s)
'''