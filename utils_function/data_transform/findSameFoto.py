from unicodedata import name
import cv2
import os
import glob
import numpy as np
from PIL import Image,ImageChops
import pandas as pd
import tqdm

def GaussianMask(sizeX,sizeY,sigma=33,center = None, fix =1):
    x = np.arange(0, sizeX, 1, float)
    y = np.arange(0, sizeY, 1, float)
    x, y = np.meshgrid(x,y) 
    if center is None:
        x0 = sizeX // 2
        y0 = sizeY // 2
    else:
        if np.isnan(center[0])==False and np.isnan(center[1])==False:            
            x0 = center[0]
            y0 = center[1]        
        else:
            return np.zeros((sizeY,sizeX))

    return fix*np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / sigma**2)

def find_same_foto(datapath):
    label_path = datapath + "/label/"
    data_path = datapath + '/orignal/'
    dataList = os.listdir(data_path)
    delList = []
    for i in tqdm.tqdm(range(len(dataList))):
        if dataList[i] not in delList:
            iterIm1 = Image.open(data_path + dataList[i])
            sameList = []
            sameList.append(dataList[i])
            for j in range(i+1,len(dataList)):
                if dataList[j] not in delList:
                    iterIm2 = Image.open(data_path + dataList[j])
                    if iterIm1.size == iterIm2.size:
                        try:
                            diff = ImageChops.difference(iterIm1,iterIm2)
                            if diff.getbbox() is None:
                                delList.append(dataList[j])
                                sameList.append(dataList[j])
                                # plot_new_heatmap(datapath,dataList[i],dataList[j])
                        except:
                            continue
            if len(sameList) > 1:
                print("\nThey r same" + str(sameList))
                # plot_new_heatmap(datapath,sameList)
                plot_new_eyevido_heatmap(datapath,sameList)


def plot_new_heatmap(path,fileList):

    img = cv2.imread(path + '/orignal/' + fileList[0])
    h, w, _ = img.shape
    heatmap = np.zeros((h,w),np.float32)
    saveName = 'A'
    for iterName in fileList:
        iterFileName = 'p' + iterName.strip('_cut.png')
        csvName = iterFileName + '-gaze.csv'
        saveName = saveName + '_' + iterFileName
        df = pd.read_csv(path + '/' +csvName)
        for index,row in df.iterrows():
            xPos = int(row['x'])
            yPos = int(row['y'])
            heatmap += GaussianMask(w, h, 33,(xPos,yPos),1)

    for i in range(len(fileList)):
        # os.remove(path + '/orignal/' + fileList[i])
        os.remove(path + '/orignal/' + fileList[i])
        prefixName = 'p' + fileList[i].strip('_cut.png')
        os.remove(path + '/label/' + prefixName + '_heatmap.png')
        os.remove(path + '/label/' + prefixName + '_pureheat.png')
        

    
    heatmap /= np.amax(heatmap)
    heatmap *= 255
    heatmap = heatmap.astype('uint8')

    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    mask = np.where(heatmap<=10, 1, 0)
    mask = np.reshape(mask, (h, w, 1))
    mask = np.repeat(mask, 3, axis=2)

    marge = img*mask + heatmap_color*(1-mask)
    marge = marge.astype("uint8")

    marge = cv2.addWeighted(img, 0.5, marge,0.5,0)
    
    # saveName = os.path.splitext(fileName1)[0] + '-' + os.path.splitext(fileName2)[0]
    pureheatSaveName = path + '/label/' + saveName + '_pureheat.png'
    heatmapSaveName = path + '/label/' + saveName + '_heatmap.png'
    cutPhotoSaveName = path + '/orignal/' + saveName + '_cut.png'

    heatmap = cv2.applyColorMap(heatmap,cv2.COLORMAP_JET)

    cv2.imwrite(pureheatSaveName,heatmap)
    cv2.imwrite(heatmapSaveName,marge)
    cv2.imwrite(cutPhotoSaveName,img)


def plot_new_eyevido_heatmap(path,fileList):

    img = cv2.imread(path + '/orignal/' + fileList[0])
    h, w, _ = img.shape
    heatmap = np.zeros((h,w),np.float32)
    saveName = 'A'
    for iterName in fileList:
        iterCSVName = iterName.split('-')[0] + '.csv'
        iterScreenshotName = iterName.split('-')[1]
        saveName = saveName + '-' + iterCSVName + iterScreenshotName
        df = pd.read_csv(path + '/Fixations/' + iterCSVName)
        df = df[df['screenshot'] == iterScreenshotName]
        minDuration = df['duration_x'].min()
        maxDuration = df['duration_x'].max()
        for index,row in df.iterrows():
            xPos = int(row['x']) + int(row['scrollPositionX'])
            yPos = int(row['y']) + int(row['scrollPositionY'])
            duration = (int(row['duration_x']) - minDuration + 0.1)/(maxDuration - minDuration + 0.1)
            heatmap += GaussianMask(w, h, 100*duration,(xPos,yPos),1)

    for i in range(len(fileList)):
        os.remove(path + '/orignal/' + fileList[i])
        # print(path + '/orignal/' + fileList[i])
        # os.remove(path + '/orignal/' + fileList[i])
        prefixName = os.path.splitext(fileList[i])[0]
        print(path + '/label/' + prefixName + '_heatmap.png')
        os.remove(path + '/label/' + prefixName + '_heatmap.png')
        os.remove(path + '/label/' + prefixName + '_pureheat.png')
        

    
    heatmap /= np.amax(heatmap)
    heatmap *= 255
    heatmap = heatmap.astype('uint8')

    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    mask = np.where(heatmap<=10, 1, 0)
    mask = np.reshape(mask, (h, w, 1))
    mask = np.repeat(mask, 3, axis=2)

    marge = img*mask + heatmap_color*(1-mask)
    marge = marge.astype("uint8")

    marge = cv2.addWeighted(img, 0.5, marge,0.5,0)
    
    # saveName = os.path.splitext(fileName1)[0] + '-' + os.path.splitext(fileName2)[0]
    pureheatSaveName = path + '/label/' + saveName + '_pureheat.png'
    heatmapSaveName = path + '/label/' + saveName + '_heatmap.png'
    cutPhotoSaveName = path + '/orignal/' + saveName + '.png'

    heatmap = cv2.applyColorMap(heatmap,cv2.COLORMAP_JET)

    # cv2.imshow('pureheatmap',heatmap)
    # cv2.imshow('heatmap',marge)
    # cv2.waitKey(0)

    cv2.imwrite(pureheatSaveName,heatmap)
    cv2.imwrite(heatmapSaveName,marge)
    cv2.imwrite(cutPhotoSaveName,img)


if __name__ == "__main__":
    _datapath = 'E:/My_MA/EyeVido/'
    datasetFolderList = os.listdir(_datapath)
    # for iterName in datasetFolderList:
    #     print(_datapath + iterName + '/shots/')
    #     find_same_foto(_datapath + iterName + '/shots/')
    find_same_foto(_datapath)