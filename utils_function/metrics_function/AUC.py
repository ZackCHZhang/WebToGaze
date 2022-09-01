

import time
from sklearn.metrics import roc_auc_score,roc_curve
import cv2
import numpy as np
import tqdm
import matplotlib.pyplot as plt
import os
import torch
from numba import jit,float32


# from matplotlib import colormaps # colormaps['jet'], colormaps['turbo']
from matplotlib.colors import LinearSegmentedColormap
from matplotlib._cm import _jet_data


def convert_jet_to_grey(img):
    (height, width) = img.shape[:2]

    cm = LinearSegmentedColormap("jet", _jet_data, N=2 ** 8)
    # cm = colormaps['turbo'] swap with jet if you use turbo colormap instead

    cm._init()  # Must be called first. cm._lut data field created here

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    fm = cv2.FlannBasedMatcher(index_params, search_params)

    # JET, BGR order, excluding special palette values (>= 256)
    fm.add(255 * np.float32([cm._lut[:256, (2, 1, 0)]]))  # jet
    fm.train()

    # look up all pixels
    query = img.reshape((-1, 3)).astype(np.float32)
    matches = fm.match(query)

    # statistics: `result` is palette indices ("grayscale image")
    output = np.uint16([m.trainIdx for m in matches]).reshape(height, width)
    result = np.where(output < 256, output, 0).astype(np.uint8)
    # dist = np.uint8([m.distance for m in matches]).reshape(height, width)

    return result  # , dist uncomment if you wish accuracy image

                            



    

# y_true = cv2.imread(_label_path)
# y_pred = cv2.imread(_pred_path)


# gray_values = np.arange(256, dtype=np.uint8)
# color_values = map(tuple, cv2.applyColorMap(gray_values, cv2.COLORMAP_JET).reshape(256, 3))
# color_to_gray_map = dict(zip(color_values, gray_values))

# y_pred = cv2.cvtColor(y_pred,cv2.COLOR_BGR2GRAY)



# y_pred = cv2.threshold(y_pred,127,1,cv2.THRESH_BINARY)

# y_true = np.apply_along_axis(color_to_gray, 2, y_true)
# y_true = cv2.threshold(y_true,127,1,cv2.THRESH_BINARY)

# print(y_pred)
# plt.imshow(y_true[1])
# y_true = np.apply_along_axis(lambda bgr: color_to_gray_map[tuple(bgr)], 2, y_true)


@jit(nopython=True)
def normalize(x, method='standard', axis=None):
    '''Normalizes the input with specified method.
    Parameters
    ----------
    x : array-like
    method : string, optional
        Valid values for method are:
        - 'standard': mean=0, std=1
        - 'range': min=0, max=1
        - 'sum': sum=1
    axis : int, optional
        Axis perpendicular to which array is sliced and normalized.
        If None, array is flattened and normalized.
    Returns
    -------
    res : numpy.ndarray
        Normalized array.
    '''
    # TODO: Prevent divided by zero if the map is flat
    # x = np.array(x, copy=False)
    if axis is not None:
        y = np.rollaxis(x, axis).reshape([x.shape[axis], -1])
        shape = np.ones(len(x.shape))
        shape[axis] = x.shape[axis]
        if method == 'standard':
            res = (x - np.mean(y, axis=1).reshape(shape)) / np.std(y, axis=1).reshape(shape)
        elif method == 'range':
            res = (x - np.min(y, axis=1).reshape(shape)) / (np.max(y, axis=1) - np.min(y, axis=1)).reshape(shape)
        elif method == 'sum':
            res = x / np.float_(np.sum(y, axis=1).reshape(shape))
        else:
            raise ValueError('method not in {"standard", "range", "sum"}')
    else:
        if method == 'standard':
            res = (x - np.mean(x)) / np.std(x)
        elif method == 'range':
            res = (x - np.min(x)) / (np.max(x) - np.min(x))
        elif method == 'sum':
            res = x / float(np.sum(x))
        else:
            raise ValueError('method not in {"standard", "range", "sum"}')
    return res

@jit(nopython=True)
def calc_loop(thresholds, S, tp, fp, n_pixels, n_fix):
    for k, thresh in enumerate(thresholds):
        above_th = np.sum(S >= thresh) # Total number of saliency map values above threshold
        tp[k+1] = (k + 1) / float(n_fix) # Ratio saliency map values at fixation locations above threshold
        fp[k+1] = (above_th - k - 1) / float(n_pixels - n_fix) # Ratio other saliency map values above threshold
    return np.trapz(tp, fp) # y, x


# roc_curve(y_true=y_true[1], y_score=y_pred[1])
def AUC_Judd(saliency_map, fixation_map, jitter=True):
    '''
    AUC stands for Area Under ROC Curve.
    This measures how well the saliency map of an image predicts the ground truth human fixations on the image.
    ROC curve is created by sweeping through threshold values
    determined by range of saliency map values at fixation locations.
    True positive (tp) rate correspond to the ratio of saliency map values above threshold
    at fixation locations to the total number of fixation locations.
    False positive (fp) rate correspond to the ratio of saliency map values above threshold
    at all other locations to the total number of possible other locations (non-fixated image pixels).
    AUC=0.5 is chance level.
    Parameters
    ----------
    saliency_map : real-valued matrix
    fixation_map : binary matrix
        Human fixation map.
    jitter : boolean, optional
        If True (default), a small random number would be added to each pixel of the saliency map.
        Jitter saliency maps that come from saliency models that have a lot of zero values.
        If the saliency map is made with a Gaussian then it does not need to be jittered
        as the values vary and there is not a large patch of the same value.
        In fact, jittering breaks the ordering in the small values!
    Returns
    -------
    AUC : float, between [0,1]
    '''
    saliency_map = np.array(saliency_map, copy=False)
    fixation_map = np.array(fixation_map, copy=False) > 0.5
    

    # If there are no fixation to predict, return NaN
    if not np.any(fixation_map):
        print('no fixation to predict')
        return np.nan
    # Make the saliency_map the size of the fixation_map
    # if saliency_map.shape != fixation_map.shape:
    # 	saliency_map = resize(saliency_map, fixation_map.shape, order=3, mode='nearest')
    # # Jitter the saliency map slightly to disrupt ties of the same saliency value
    # if jitter:
    # 	saliency_map += random.rand(*saliency_map.shape) * 1e-7
    # Normalize saliency map to have values between [0,1]
    # saliency_map = normalize(saliency_map, method='range')
    # fixation_map = normalize(fixation_map, method='range')
    # fixation_map = fixation_map >0,5

    S = saliency_map.ravel()
    F = fixation_map.ravel()
    S_fix = S[F] # Saliency map values at fixation locations
    n_fix = len(S_fix)
    n_pixels = len(S)
    # Calculate AUC
    thresholds = sorted(S_fix, reverse=True)
    tp = np.zeros(len(thresholds)+2)
    fp = np.zeros(len(thresholds)+2)
    tp[0] = 0; tp[-1] = 1
    fp[0] = 0; fp[-1] = 1
    thresholds = np.array(thresholds)
    res = calc_loop(thresholds, S, tp, fp, n_pixels, n_fix)
    return res
    # end = time.time()
    # print(end - start)
    # print(res)
    # for k, thresh in enumerate(thresholds):
    #     above_th = np.sum(S >= thresh) # Total number of saliency map values above threshold
    #     tp[k+1] = (k + 1) / float(n_fix) # Ratio saliency map values at fixation locations above threshold
    #     fp[k+1] = (above_th - k - 1) / float(n_pixels - n_fix) # Ratio other saliency map values above threshold
    # start = time.time()
    # print(start-end)
    # print(np.trapz(tp, fp))
    # return np.trapz(tp, fp) # y, x

# print(AUC_Judd(y_pred,y_true))

if __name__ == "__main__":
    _pred_path = "E:/My_MA/Results/paper_rebuild/gaze_new/36/"
    _pred_path = "E:/My_MA/Results/FIWI/otherModal/salgan/"
    # _label_path = "E:/My_MA/new_gaze/Test/label/"
    _label_path = "E:/My_MA/FIWItest/label/"
    # _label_path = "E:/My_MA/EyevedoDataset/test/label/"
    _log_path = 'E:/My_MA/Results/FIWI/otherModal/salgan/AUC.txt'
    _root_path = "E:/My_MA/Results/GazeMiningResult/otherModel/"
    all_dir_flag = False

    if all_dir_flag:
        dir_list = os.listdir(_root_path)
        for itDir in dir_list:
            if os.path.isdir(_root_path + itDir):
                predict_list = os.listdir(_root_path + itDir)
                lable_list = os.listdir(_label_path)
                real_label = []
                for iter in lable_list:
                    iterName = iter.split('_')[-1]
                    if iterName == "pureheat.png":
                        real_label.append(iter)
                
                _log_path = _root_path + itDir + "/AUC.txt"
                
                sumMae = 0.0
                sumCC = 0.0
                sumNSS = 0.0
                sumAUC = 0.0
                count = 0
                # if os.path.exists(_log_path):
                #     print(_log_path)
                #     print("Already Calced")
                # else:

                f = open(_log_path,'w')
                exceptList = []

                for i in tqdm.tqdm(range(len(real_label))):
                    
                    name = os.path.splitext(real_label[i])[0]
                    # nameList = name.split('_')
                    # predName = nameList[0] + '_'
                    # for j in range(1,3):
                    #     predName +=  nameList[j] + '_'
                    # predName = predName + '.jpeg'
                    # predName = nameList[0] + '.png'
                    print(name)

                    predName = real_label[i].replace('_pureheat.png','.png')
                    print(predName)

                    y_true = cv2.imread(_label_path + real_label[i])
                    y_pred = cv2.imread(os.path.join(_root_path, itDir, predName))
                    # print(_root_path + itDir + predName)
                    # y_true = convert_jet_to_grey(y_true)
                    
                    y_pred = cv2.cvtColor(y_pred,cv2.COLOR_BGR2GRAY)
                    y_true = cv2.cvtColor(y_true,cv2.COLOR_BGR2GRAY)
                    # cv2.imshow("y_pred",y_pred)
                    # cv2.imshow("y_true",y_true)
                    # cv2.waitKey(0)

                    y_pred = (y_pred -np.min(y_pred))/(np.max(y_pred) - np.min(y_pred))
                    y_true = (y_true -np.min(y_true))/(np.max(y_true) - np.min(y_true))
                    if y_pred.shape != y_true.shape:
                        y_true = cv2.resize(src=y_true,dst=y_true,dsize=(y_pred.shape[1],y_pred.shape[0]))
                        print(y_true.shape)
                        print(y_pred.shape)
                    if y_pred.shape[0] > 3000 or y_pred.shape[1]>3000:
                        exceptList.append(_root_path + itDir + predName + '\n')
                    else:
                        # print("SUM:",sumVal)
                        y_true = np.array(y_true)
                        y_pred = np.array(y_pred)
                        iterAUC = AUC_Judd(y_pred, y_true)
                        print("\n AUC:",iterAUC)
                        f.write("\n AUC:" + str(iterAUC))
                        sumAUC += iterAUC
                print(sumAUC/len(real_label))
                f.write("\n" + str(sumAUC/len(real_label)))
                f.write("\n" + "those file is too big to hard calc auc" + str(exceptList) + '\n' + 'round' + str(len(exceptList)/len(real_label)))
                f.close()
                

    else:


        predict_list = os.listdir(_pred_path)
        lable_list = os.listdir(_label_path)
        real_label = []
        for iter in lable_list:
            iterName = iter.split('_')[-1]
            if iterName == "pureheat.png":
                real_label.append(iter)

        sumAUC = 0.0

        f = open(_log_path,'w')

        # y_true_all = np.empty(shape=[0,1])
        # y_pred_all = np.empty(shape=[0,1])
        # for i in range(len(real_label)):
        #     name = os.path.splitext(real_label[i])[0]
        #     nameList = name.split('_')
        #     predName = nameList[0] + '_'
        #     for j in range(1,3):
        #         predName +=  nameList[j] + '_'
        #     predName = predName + 'cut.jpeg'

        #     y_true = cv2.imread(_label_path + real_label[i])
        #     y_pred = cv2.imread(_pred_path + predName)
        #     y_pred = cv2.cvtColor(y_pred,cv2.COLOR_BGR2GRAY)
        #     y_true = cv2.cvtColor(y_true,cv2.COLOR_BGR2GRAY)
        #     y_true_all_iter = np.array(y_true).ravel()
        #     y_pred_all_iter = np.array(y_pred).ravel()
        #     y_true_all = np.append(y_true_all,y_true_all_iter)
        #     y_pred_all = np.append(y_pred_all,y_pred_all_iter)
        # AUC_aver = AUC_Judd(y_pred_all,y_true_all)
        # print("Total AUC is:", AUC_aver)
        # f.write("\n Total AUC is :" + str(AUC_aver))
        exceptList = []

        for i in tqdm.tqdm(range(len(real_label))):
            
            name = os.path.splitext(real_label[i])[0]
            nameList = name.split('_')
            predName = nameList[0] + '_'
            predName = real_label[i].replace('_pureheat.png','.jpg')
            print(predName)

            y_true = cv2.imread(_label_path + real_label[i])
            # y_true = convert_jet_to_grey(y_true)
            y_pred = cv2.imread(_pred_path + predName)
            y_pred = cv2.cvtColor(y_pred,cv2.COLOR_BGR2GRAY)
            y_true = cv2.cvtColor(y_true,cv2.COLOR_BGR2GRAY)
            # cv2.imshow("Ytrue", y_true)
            # cv2.imshow("Y-pred", y_pred)
            # cv2.waitKey(0)
            if y_pred.shape != y_true.shape:
                y_true = cv2.resize(src=y_true,dst=y_true,dsize=(y_pred.shape[1],y_pred.shape[0]))
                print(y_true.shape)
                print(y_pred.shape)

            if y_pred.shape[0] > 3000 or y_pred.shape[1]>3000:
                exceptList.append(_pred_path + predName + '\n')
            else:

                print(_label_path + real_label[i])
                print(_pred_path + predName)
                y_pred = normalize(y_pred,method='range')
                y_true = normalize(y_true, method='range')
                f.write(predName)
                y_true = np.array(y_true)
                y_pred = np.array(y_pred)
                iterAUC = AUC_Judd(y_pred, y_true)
                print("\n AUC:",iterAUC)
                f.write("\n AUC:" + str(iterAUC))
                sumAUC += iterAUC
        print(sumAUC/len(real_label))
        f.write("\n" + str(sumAUC/len(real_label)))
        f.write("\n" + "those file is too big to hard calc auc" + str(exceptList) + '\n' + 'round' + str(len(exceptList)/len(real_label)))
        f.close()
        