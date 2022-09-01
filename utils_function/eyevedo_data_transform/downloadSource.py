import os
from turtle import shape
import pandas as pd
import wget
import csv
import numpy as np
import cv2
import tqdm

def GaussianMask(sizeX,sizeY,sigma=33,center = None, fix =1):
    '''
    this function is a gussianmask:
    @param: sizeX: the img x size
            sizeY: the img y size
            sigma: sigma for gaussen
            center: gaussian mean
            fix: gaussian max
    @return: mask of gaussian
    '''
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



def downloadCSV(settingfolder,csv_folder,prefix,destination):
    '''
        Use the fuc build the orignal snapshot download url and download them
        @Author: Ciheng Zhang
        @params:
            settingfolder: the folder save some settingfiles(relationship between web_id and study_id)
            csv_folder: the folder include all the csv file(F_Ux_Sx.csv)
            prefix: source url prefix of eyecedo
            destination: which folder aims to save orignal figures
        @return: null
    '''
    settings = {}
    with open(settingfolder + 'webs.csv', 'r') as csvFile:
        csvsettings = csv.reader(csvFile)
        settings = {row[1]:row[0] for row in csvsettings}
    print(settings)
    
    fileList = os.listdir(csv_folder)
    allDict = {}
    for iter in fileList:
        if iter.endswith(".csv"):
            print(iter)
            downloadList = []
            df = pd.read_csv(csv_folder + iter)
            if len(df) > 1:
                for i in range(len(df)):
                    iterStudyId = settings[str(df["web_id"][i])]
                    iterUrl = prefix + "/study_" + str(iterStudyId) + "/web_" + str(df["web_id"][i]) + "/" + str(df["screenshot"][i])
                    if iterUrl not in downloadList:
                        downloadList.append(iterUrl)
                    # print("\033[1;35m start dwonload from {}\033[0m".format(iterUrl))
            allDict.update({iter:downloadList})                
    # print(allDict)
    for key in allDict:
        for iter in allDict[key]:
            # print(destination + '/' + key.split('.')[0] + '-' + iter.split('/')[-1])
            wget.download(iter,destination + '/' + key.split('.')[0] + '-' + iter.split('/')[-1])



        



def plotHeatmap(path, alpha = 0.5, threshold = 10):
    '''
        Use this function to draw the eye heatmap
        @param: 
                path: the root folder where include the cvsfiles,
                (datafolder named orignal for jpgs, 
                 csv folder for csvfiles named Fixations, 
                 setting folder for webs.csv which is export from sql include the relation bewteen study_id and web_id)
                alpha: merge rate imgfile and heatmap
                threshold: heatmap threshold(0-255)
        @return heatmap
    '''
    print("\033[1;35m start heatmap generation!\033[0m")
    imgNameList = []
    datapath = path + '/orignal/'
    csvpath = path + '/Fixations/'
    savePath = path + '/label/'
    for iter in os.listdir(datapath):
        if iter.endswith('.jpg'):
            imgNameList.append(iter)
    # 这里要加上读出文件LIST

    for iterImg in tqdm.tqdm(imgNameList):
    # 这里读取长宽
    # 判断是不是已经处理完成
        judgName = iterImg.split('.')[0] + '_pureheat.png' 
        if judgName not in os.listdir(savePath):

            img = cv2.imread(datapath + iterImg)
            size = img.shape
            w = size[1]
            h = size[0]

            heatmap = np.zeros((h,w),np.float32)

            # 这里要改成需要的文件名字
            csvName = os.path.splitext(iterImg)[0]
            csvName = csvName.split('-')[0]
            screenshotName = os.path.splitext(iterImg)[0].split('-')[1] + '.jpg'
            csvName = csvName + '.csv'
            df = pd.read_csv(csvpath + csvName)

            # print(df[df['screenshot'] == screenshotName])
            maxDuration = int(df[df['screenshot'] == screenshotName]['duration_x'].max()) 
            minDuration = int(df[df['screenshot'] == screenshotName]['duration_x'].min()) 
            # maxDuration = int(df['duration_x'].max()) 
            # minDuration = int(df['duration_x'].min()) 

            for index,row in df.iterrows():
                if row['screenshot'] == screenshotName:
                    xPos = int(row['x']) + int(row['scrollPositionX'])
                    yPos = int(row['y']) + int(row['scrollPositionY'])
                    duration = (int(row['duration_x']) - minDuration + 0.1)/(maxDuration - minDuration + 0.1)
                    heatmap += GaussianMask(w, h, 100 * duration,(xPos,yPos),1) 

            heatmap /= np.amax(heatmap)
            heatmap *= 255
            heatmap = heatmap.astype('uint8')

            heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

            mask = np.where(heatmap<=threshold, 1, 0)
            mask = np.reshape(mask, (h, w, 1))
            mask = np.repeat(mask, 3, axis=2)

            # Marge images
            marge = img*mask + heatmap_color*(1-mask)
            marge = marge.astype("uint8")
            # heatmap = cv2.applyColorMap(heatmap,cv2.COLORMAP_JET)
            # cv2.imshow('demo',heatmap)
            marge = cv2.addWeighted(img, 1-alpha, marge,alpha,0)

            saveName = iterImg.split('.')[0]
            # print("saveName:"+saveName)

            pureheatSaveName = savePath + saveName + '_pureheat.png'
            heatmapSaveName = savePath  + saveName + '_heatmap.png'
            # cutPhotoSaveName = savePath + '/gaze-image/' + saveName + '_cut.png'
            
            heatmap = cv2.applyColorMap(heatmap,cv2.COLORMAP_JET)
            # cv2.imshow("pureheat", heatmap)
            # cv2.waitKey(0)
            # cv2.imshow("heatmap", marge)
            # cv2.waitKey(0)
            
            cv2.imwrite(pureheatSaveName,heatmap)
            cv2.imwrite(heatmapSaveName,marge)
            # cv2.imwrite(cutPhotoSaveName,img)
        
    print("\033[1;35m generation finished!!\033[0m")

            



if __name__ == "__main__":
    _schema = "https://eyevido.de/portal/img/user_uploads/user_913577351/study_[study_id]/web_[web_id]/[ table “web_results” column “screenshot”]"
    _prefix = "https://eyevido.de/portal/img/user_uploads/user_913577351"
    _csv_folder = "E:/My_MA/EyeVido/Fixations/"
    _setting_folder = "E:/My_MA/EyeVido/Settings/"
    _destination_foler = "E:/My_MA/EyeVido/orignal/"
    _path = "E:/My_MA/EyeVido/"
    downloadCSV(settingfolder=_setting_folder,csv_folder= _csv_folder, prefix=_prefix, destination=_destination_foler)
    plotHeatmap(path=_path)

