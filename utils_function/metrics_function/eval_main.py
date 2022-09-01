from cgi import test
import os
import skimage
import NSS
import cv2
import numpy as np
from sklearn.metrics import mean_absolute_error
from matplotlib.colors import LinearSegmentedColormap
from matplotlib._cm import _jet_data
import tqdm

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

if __name__ == '__main__':
    all_dir_flag = False

    # _pred_path = "E:/My_MA/Results/GazeMiningResult/asppupdate/58/"
    _pred_path = "E:/My_MA/Results/paper_rebuild/gaze_new/36/"
    # _pred_path = "E:/My_MA/Results/FIWI/images/"
    _label_path = "E:/My_MA/new_gaze/Test/label/"
    # _label_path = "E:/My_MA/FIWItest/label/"
    # _label_path = "E:/My_MA/EyevedoDataset/test/label/"
    _log_path = _pred_path + "NSS_CC_MAE.txt"
    # _log_path = 'E:/My_MA/Results/EyeVidoResult/asppupdate/55/gan_alldata.txt'
    _root_path = "E:/My_MA/Results/FIWI/otherModal/"

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
                
                _log_path = _root_path + itDir + "/NSS_CC_MAE.txt"
                
                sumMae = 0.0
                sumCC = 0.0
                sumNSS = 0.0
                count = 0
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
                    if y_pred is None:
                        predName = real_label[i].replace('_pureheat.png','.jpg')
                        print(itDir)
                        y_pred = cv2.imread(os.path.join(_root_path, itDir, predName))
                    
                    if y_pred is not None:
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
                            count += 1
                            iterNSS = NSS.calc_NSS(y_pred, y_true)
                            iterMAE = mean_absolute_error(y_pred,y_true)
                            iterCC = NSS.CC(y_pred,y_true)
                            f.write('\n' + real_label[i] + "\nNSS:" + str(iterNSS) + '\nCC:' + str(iterCC) + '\nMAE:' + str(iterMAE))
                            sumNSS += iterNSS
                            sumMae += iterMAE
                            sumCC += iterCC
                            # else:
                            #     print(real_label[i])
                    else:
                        f.write("\n NO SUCH FILE: " + predName)
                        print("\n NO SUCH FILE: " + predName)
                print(sumCC/count)
                f.write("\nTOTAL:" + "\n CC:" + str(sumCC/count) + "\n MAE:" + str(sumMae/count) + "\n NSS:" + str(sumNSS/count))
                f.write("\n" + "those file is too big to hard calc auc" + str(exceptList) + '\n' + 'round' + str(len(exceptList)/count))
                f.close()

    else:
        predict_list = os.listdir(_pred_path)
        lable_list = os.listdir(_label_path)
        real_label = []
        for iter in lable_list:
            iterName = iter.split('_')[-1]
            if iterName == "pureheat.png":
                real_label.append(iter)
        
        sumMae = 0.0
        sumCC = 0.0
        sumNSS = 0.0
        count = 0
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

            predName = real_label[i].replace('_pureheat.png','_cut.jpeg')
            print(predName)

            y_true = cv2.imread(_label_path + real_label[i])
            y_pred = cv2.imread(_pred_path + predName)
            print(_pred_path + predName)
            # y_true = convert_jet_to_grey(y_true)
            
            y_pred = cv2.cvtColor(y_pred,cv2.COLOR_BGR2GRAY)
            y_true = cv2.cvtColor(y_true,cv2.COLOR_BGR2GRAY)
            # cv2.imshow("y_pred",y_pred)
            # cv2.imshow("y_true",y_true)
            # cv2.waitKey(0)
            # tmp = y_pred.mean()
            # h, w = y_pred.shape
            # for row in tqdm.tqdm(range(h)):
            #     for col in range(w):
            #         it = y_pred[row][col]
            #         # img[row, col] = (255 - b, 255 - g, 255 - r)
            #         # img[row, col] = (255 - b, g, r)
            #         # img[row, col] = (255 - b, g, 255 - r)
            #         if it > tmp:
            #             itc = it- tmp
            #             y_pred[row][col] = itc
            #         else:
            #             y_pred[row][col] = 0
 

            y_pred = (y_pred -np.min(y_pred))/(np.max(y_pred) - np.min(y_pred))
            y_true = (y_true -np.min(y_true))/(np.max(y_true) - np.min(y_true))
            if y_pred.shape != y_true.shape:
                y_true = cv2.resize(src=y_true,dst=y_true,dsize=(y_pred.shape[1],y_pred.shape[0]))
                print(y_true.shape)
                print(y_pred.shape)
            if y_pred.shape[0] > 3000 or y_pred.shape[1]>3000:
                exceptList.append(_pred_path + predName + '\n')
            else:
                # print("SUM:",sumVal)
                count += 1
                iterNSS = NSS.calc_NSS(y_pred, y_true)
                iterMAE = mean_absolute_error(y_pred,y_true)
                iterCC = NSS.CC(y_pred,y_true)
                f.write('\n' + real_label[i] + "\nNSS:" + str(iterNSS) + '\nCC:' + str(iterCC) + '\nMAE:' + str(iterMAE))
                sumNSS += iterNSS
                sumMae += iterMAE
                sumCC += iterCC
                # else:
                #     print(real_label[i])
        print(sumCC/len(real_label))
        f.write("\nTOTAL:" + "\n CC:" + str(sumCC/len(real_label)) + "\n MAE:" + str(sumMae/len(real_label)) + "\n NSS:" + str(sumNSS/len(real_label)))
        f.write("\n" + "those file is too big to hard calc auc" + str(exceptList) + '\n' + 'round' + str(len(exceptList)/len(real_label)))
        f.close()