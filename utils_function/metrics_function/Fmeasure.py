import numpy as np
import os
import cv2

def meanFcal(pred, gt):
    ########################meanF##############################
    th = 1 * pred.mean()
    if th > 1:
        th = 1

    # th = 0.7

    binary = np.zeros_like(pred)
    binary[pred >= th] = 1
    hard_gt = np.zeros_like(gt)
    hard_gt[gt > 0.5] = 1
    tp = (binary * hard_gt).sum()
    if tp == 0:
        mfmeasure = 0
    else:
        sum1 = binary.sum()
        sum2 = hard_gt.sum()
        pre = tp / binary.sum()
        rec = tp / hard_gt.sum()
        mfmeasure = 1.3 * pre * rec / (0.3 * pre + rec)
    return mfmeasure

if __name__ == "__main__":
    # _pred_path = "E:/My_MA/Results/GazeMiningResult/asppupdate/58/"
    # _pred_path = "E:/My_MA/Results/EyeVidoResult/asppupdate/58/"
    _pred_path = "E:/My_MA/Results/EyeVidoResult/asppupdate/55/"
    # _label_path = "E:/My_MA/Test/label/"
    # _label_path = "E:/My_MA/FIWItest/label/"
    _label_path = "E:/My_MA/EyevedoDataset/test/label/"
    _log_path = _pred_path + "Fmeasure.txt"
    # _log_path = 'E:/My_MA/Results/EyeVidoResult/asppupdate/55/gan_alldata.txt'
    _root_path = "E:/My_MA/Results/EyeVidoResult/EyeVedoOnly/OldMM/"
    # _pred_path = "E:/My_MA/TestData/result/kroner2020_result/trained_result/saliencymap/"
    # _label_path = "E:/My_MA/TestData/label/"
    predict_list = os.listdir(_pred_path)
    lable_list = os.listdir(_label_path)
    real_label = []
    for iter in lable_list:
        iterName = iter.split('_')[-1]
        if iterName == "pureheat.png":
            real_label.append(iter)

    sumMAE = 0.0

    f = open(_log_path,'w')
    for i in range(len(real_label)):
        
        name = os.path.splitext(real_label[i])[0]
        nameList = name.split('_')
        predName = real_label[i].replace('_pureheat.png','.jpeg')
        # predName = nameList[0] + '_'
        # for j in range(1,3):
        #     predName +=  nameList[j] + '_'
        # predName = predName + '.jpeg'

        y_true = cv2.imread(_label_path + real_label[i])
        y_pred = cv2.imread(_pred_path + predName)
        y_pred = cv2.cvtColor(y_pred,cv2.COLOR_BGR2GRAY)
        y_true = cv2.cvtColor(y_true,cv2.COLOR_BGR2GRAY)
        y_pred = (y_pred -np.min(y_pred))/(np.max(y_pred) - np.min(y_pred))
        y_true = (y_true -np.min(y_true))/(np.max(y_true) - np.min(y_true))
        print("SUM:",sumMAE)
        iterFM = meanFcal(y_pred,y_true)
        print("Fmeasure:", iterFM)
        f.write("\n FM:" + str(iterFM))
        sumMAE += iterFM
    print(sumMAE/len(real_label))
    f.write("\n" + str(sumMAE/len(real_label)))
    f.close()