import os
import cv2
from numpy.lib.function_base import delete
import pandas
from tqdm import tqdm
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
'''
 This class is use to load gaze dataset and plot the csv and image
 @Author: zch
'''
class EyegazeDataloader():
    '''
    init the class
    @Author: Ciheng Zhang
    @Paramater: datapath, the root path of eyegaze dataset
    '''
    def __init__(self, datapath):
        self.datapath = datapath
        self.imgNameList = []
        self.suffixList = ['blind','gaze','mouse']
        self.pointSize = 1
        self.pointColor = (0,0,255)
        self.thickness = 4
    
    '''
    iteration the dir and save all name of screenshot in list
    '''
    def gotImagesName(self):
        for imgName in os.listdir(self.datapath):
            iterName = os.path.splitext(imgName)[0]
            photoNameList = iterName.split('_')
            photoName = photoNameList[len(photoNameList)-1]
            if imgName.endswith('png') and photoName != "after" and photoName != "heatmap" and photoName != "pureheat":
                juggName = iterName + "_heatmap.png"
                print(self.datapath+'/gaze-mask/'+ iterName + "_heatmap.png")
                if not os.path.isfile(self.datapath+'/gaze-mask/'+ iterName + "_heatmap.png"):
                    self.imgNameList.append(self.datapath+'/'+imgName)

        if not os.path.exists(self.datapath + "/gaze-image"):
            os.mkdir(self.datapath + "/gaze-image")

        if not os.path.exists(self.datapath + "/gaze-mask"):
            os.mkdir(self.datapath + "/gaze-mask")
                
    
    '''
    draw the points on the image
    @param: suffix: "blind", "gaze", "mouse"
    '''
    def plotGaze(self,suffix):
        if self.imgNameList == []:
            self.gotImagesName
        if suffix not in self.suffixList:
            print ('worng suffix name')
            quit()

        for iterImg in self.imgNameList:
            print(iterImg)
            img = cv2.imread(iterImg)
            
            csvName = os.path.splitext(iterImg)[0]
            csvName = csvName + '-' + suffix + '.csv'
            df = pandas.read_csv(csvName)
            for index,row in df.iterrows():
                xPos = int(row['x'])
                yPos = int(row['y'])
                cv2.circle(img,(xPos,yPos),self.pointSize,self.pointColor)

            saveName = os.path.splitext(iterImg)[0]
            saveName = saveName + '_after.png'
            print(saveName)
            cv2.imwrite(saveName,img)

    '''
        Use this function to dran the eye heatmap
        @para: alpha: merge rate imgfile and heatmap
               threshold: heatmap threshold(0-255)
               suffix: gaze,mouse,blind
        @return heatmap
    '''
    def plotHeatmap(self, alpha = 0.5, threshold = 10, suffix='gaze'):
        if self.imgNameList == []:
            self.gotImagesName
        if suffix not in self.suffixList:
            print ('worng suffix name')
            quit()
        print(len(self.imgNameList))

        for iterImg in tqdm(self.imgNameList):
        
            print(np.array(iterImg))
            img = cv2.imread(iterImg)

            l,r,b,t = self.remove_black_edges(img)

            h, w, _= img.shape
            print("L:",l)
            print("r:",r)
            print("b:",b)
            print("t:",t)
            print(h)
            print(w)
            if r-l > 300 and t-b > 300:
                heatmap = np.zeros((h,w),np.float32)
                csvName = os.path.splitext(iterImg)[0]
                csvName = csvName + '-' + suffix + '.csv'
                df = pandas.read_csv(csvName)
                for index,row in tqdm(df.iterrows()):
                    xPos = int(row['x'])
                    yPos = int(row['y'])
                    heatmap += self.GaussianMask(w, h, 33,(xPos,yPos),1) 

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

                savePath = os.path.split(iterImg)[0]
                saveName = os.path.split(iterImg)[1]
                saveName = saveName.split('.')[0]
                print("saveName:"+saveName)

                pureheatSaveName = savePath + '/gaze-mask/' + saveName + '_pureheat.png'
                heatmapSaveName = savePath + '/gaze-mask/' + saveName + '_heatmap.png'
                cutPhotoSaveName = savePath + '/gaze-image/' + saveName + '_cut.png'
                
                heatmap = cv2.applyColorMap(heatmap,cv2.COLORMAP_JET)

                if b > 50 or w-r > 50:
                    if b < 50:
                        b = 0
                    heatmap = heatmap[b:t, l:r]
                    marge = marge[b:t,l:r]
                    img = img[b:t,l:r]
                
                cv2.imwrite(pureheatSaveName,heatmap)
                cv2.imwrite(heatmapSaveName,marge)
                cv2.imwrite(cutPhotoSaveName,img)
            

        
    
    '''
        this function is a gussianmask:
        @param: sizeX: the img x size
                sizeY: the img y size
                sigma: sigma for gaussen
                center: gaussian mean
                fix: gaussian max
        @return: mask of gaussian
    '''
    def GaussianMask(self,sizeX,sizeY,sigma=33,center = None, fix =1):
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

    '''
        this function use to delet all figure with suffix 'gaze', 'heatmap', 'pureheat'
    '''
    def deleteAll(self):        
        for imgName in os.listdir(self.datapath + "/gaze-image"):
            iterName = os.path.splitext(imgName)[0]
            photoNameList = iterName.split('_')
            photoName = photoNameList[len(photoNameList)-1]
            if imgName.endswith('png'):
                if photoName == "cut" :
                    os.remove(self.datapath + '/gaze-image/'+ imgName)
                
        for imgName in os.listdir(self.datapath + "/gaze-mask"):
            iterName = os.path.splitext(imgName)[0]
            photoNameList = iterName.split('_')
            photoName = photoNameList[len(photoNameList)-1]
            if imgName.endswith('png'):
                if photoName == "after" or photoName == "heatmap" or photoName == "pureheat":
                    os.remove(self.datapath +'/gaze-mask/' + imgName)

        for imgName in os.listdir(self.datapath ):
            iterName = os.path.splitext(imgName)[0]
            photoNameList = iterName.split('_')
            photoName = photoNameList[len(photoNameList)-1]
            if imgName.endswith('png'):
                if photoName == "after" or photoName == "heatmap" or photoName == "pureheat":
                    os.remove(self.datapath +'/' + imgName)

    def remove_black_edges(self, img):
        gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        x=gray.shape[1]
        y=gray.shape[0]
        edges_x=[]
        edges_y=[]

        #optimization...        
        for i in range(x-1):
            for j in range(y-1):
                if int(gray[j][i])==255 and int(gray[j+1][i] == 255 and int(gray[j][i+1]) == 255 and int(gray[j+1][i+1]) == 255):
                    edges_x.append(i)
                    edges_y.append(j)

        #...optimization
        if edges_x != []:
            left=min(edges_x)               #左边界
            right=max(edges_x)              #右边界
        else:
            left = 0
            right = x - 1

        if edges_y != []:
            bottom=min(edges_y)             #底部
            top=max(edges_y)                #顶部
        else:
            bottom = 0
            top = y-1

        print("l:{},r:{},b:{},t:{}".format(left,right,bottom,top))

        return left,right,bottom,top




if __name__ ==  '__main__':
    parser = argparse.ArgumentParser(description='Use it to work make points on eyegaze img')
    parser.add_argument('-datapath', type=str,help='the dataset path on your computer')
    parser.add_argument('-worktype', type=str,help='which csv your want to work blind, gaze, mouse')
    parser.add_argument('-goal', type=str,help='plot points or heatmap or delete')
    args = parser.parse_args()

    myDataLoader = EyegazeDataloader(args.datapath)
    myDataLoader.gotImagesName()
    if args.goal == 'points':
        myDataLoader.plotGaze(args.worktype)
    elif args.goal == 'heatmap':
        myDataLoader.plotHeatmap(suffix=args.worktype)
    elif args.goal == 'delete':
        myDataLoader.deleteAll()
    else:
        print("goal is wrong")