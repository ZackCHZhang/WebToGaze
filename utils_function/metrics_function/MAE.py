from sklearn.metrics import mean_absolute_error
import cv2
import numpy as np
import os

# _pred_path = "E:/My_MA/Results/EyeVidoResult/BAS/"
# _label_path = "E:/My_MA/TestData/label/p1_amazon_0_pureheat.png"

# y_pred = cv2.imread(_pred_path)
# y_true = cv2.imread(_label_path)

# y_pred = cv2.cvtColor(y_pred,cv2.COLOR_BGR2GRAY)
# y_true = cv2.cvtColor(y_true,cv2.COLOR_BGR2GRAY)
# # print(y_pred)
# # print(y_true)
# y_pred = (y_pred -np.min(y_pred))/(np.max(y_pred) - np.min(y_pred))
# y_true = (y_true -np.min(y_true))/(np.max(y_true) - np.min(y_true))

# print(mean_absolute_error(y_pred=y_pred, y_true=y_true))

if __name__ == "__main__":
    _pred_path = "E:\My_MA\Results\GazeMiningResult\Kroner_finetuning_salicon50eyevedo50/"
    _label_path = "E:/My_MA/Test/label/"
    predict_list = os.listdir(_pred_path)
    lable_list = os.listdir(_label_path)
    real_label = []
    for iter in lable_list:
        iterName = iter.split('_')[-1]
        if iterName == "pureheat.png":
            real_label.append(iter)

    sumMAE = 0.0

    f = open('E:/My_MA/Results/GazeMiningResult/Kroner_finetuning_salicon50eyevedo50/Kroner_finingtuning_MAE.txt','w')
    for i in range(len(real_label)):
        
        name = os.path.splitext(real_label[i])[0]
        # nameList = name.split('_')
        # predName = nameList[0] + '_'
        # for j in range(1,3):
        #     predName +=  nameList[j] + '_'
        # predName = predName + '.jpeg'
        # predName = nameList[0] + '.png'
        print(name)

        predName = real_label[i].replace('_pureheat.png','_cut.jpeg')
        print(predName)

        y_true = cv2.imread(_label_path + real_label[i])
        y_pred = cv2.imread(_pred_path + predName)
        y_pred = cv2.cvtColor(y_pred,cv2.COLOR_BGR2GRAY)
        y_true = cv2.cvtColor(y_true,cv2.COLOR_BGR2GRAY)

        y_pred = (y_pred -np.min(y_pred))/(np.max(y_pred) - np.min(y_pred))
        y_true = (y_true -np.min(y_true))/(np.max(y_true) - np.min(y_true))
        if y_pred.shape != y_true.shape:
            y_true = cv2.resize(src=y_true,dst=y_true,dsize=(y_pred.shape[1],y_pred.shape[0]))
            print(y_true.shape)
            print(y_pred.shape)
        print("SUM:",sumMAE)
        iterMAE = mean_absolute_error(y_pred, y_true)
        f.write(real_label[i] + "\n MAE:" + str(iterMAE) + '\n')
        sumMAE += iterMAE
        # else:
        #     print(real_label[i])
    print(sumMAE/len(predict_list))
    f.write("\n" + str(sumMAE/len(predict_list)))
    f.close()