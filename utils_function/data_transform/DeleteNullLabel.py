import os
import numpy as np
import cv2
from tqdm import tqdm

data_dir = 'E:/My_MA/Training'
tra_image_dir = '/train/'
tra_label_dir = '/label/'

def camparePixel(imArray):
    for i in range(imArray.shape[0]):
        for j in range(imArray.shape[1]):
            if imArray[i][j][0] != 128 and imArray[i][j][1] != 0 and imArray[i][j][2] != 0:
                deleteFlag = False
                return False
    return True

for iter in tqdm(os.listdir(data_dir + tra_label_dir)):
    fileName = os.path.splitext(iter)[0]
    if fileName.split('_')[-1] == 'pureheat':
        im = cv2.imread(data_dir + tra_label_dir + iter)
        imArray = np.array(im)
        deleteFlag = True
        deleteFlag =  camparePixel(imArray)
        
        if deleteFlag:
            otherRootName = iter.rstrip('_pureheat.png')
            print('\n'+otherRootName)
            pureheatName = data_dir + tra_label_dir + iter
            heatmapName = data_dir + tra_label_dir + otherRootName + '_heatmap.png'
            cutName = data_dir + tra_image_dir + otherRootName + '_cut.png'
            os.remove(pureheatName)
            os.remove(heatmapName)
            os.remove(cutName)



