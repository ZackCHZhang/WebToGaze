from posixpath import splitext
import cv2
import os
import glob
import tqdm

_path = 'E:/My_MA/EyevedoDataset_copy/train/label/'
nameList = glob.glob(_path + '*_pureheat.png')
_num_boundingbox = 2

def calc_area(elem):
    return elem[2] * elem[3]

for file in tqdm.tqdm(nameList):
    print(file)
    iterName = os.path.splitext(file)[0]
    tmpName = iterName.split('\\')[1]
    tmpNameList = tmpName.split('_')
    bboxFileName = ''
    
    for i in range(len(tmpNameList)-1):
        if i == 0:
            bboxFileName += tmpNameList[i]
        else:
            bboxFileName += '_' + tmpNameList[i]
    
    bboxFileName = _path + bboxFileName + '_bbox.txt'

    print(bboxFileName)
    with open(bboxFileName,"w") as f:
        # Grayscale then Otsu's threshold
        image = cv2.imread(file)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        # Find contours
        cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        boundingBoxList = []
        for c in cnts:
            x,y,w,h = cv2.boundingRect(c)
            boundingBoxList.append([x,y,w,h])

        boundingBoxList.sort(key=calc_area, reverse=True)
        print(boundingBoxList)
        
        for it in boundingBoxList:
            if calc_area(it) > 900:
                f.write(str(it[0]) + ';' + str(it[1]) + ';' + str(it[2]) + ';' + str(it[3]) + '\n')
                cv2.rectangle(image, (it[0], it[1]), (it[0] + it[2], it[1] + it[3]), (36,255,12), 2)

        # if _num_boundingbox > len(boundingBoxList):
        #     loopNum = len(boundingBoxList)
        # else:
        #     loopNum = _num_boundingbox
        # for i in range(loopNum):
        #     f.write(str(boundingBoxList[i][0]) + ';' + str(boundingBoxList[i][1]) + ';' + str(boundingBoxList[i][2]) + ';' + str(boundingBoxList[i][3]) + '\n')
        #     cv2.rectangle(image, (boundingBoxList[i][0], boundingBoxList[i][1]), (boundingBoxList[i][0] + boundingBoxList[i][2], boundingBoxList[i][1] + boundingBoxList[i][3]), (36,255,12), 2)

        # for c in cnts:
        #     x,y,w,h = cv2.boundingRect(c)
        #     f.write(str(x) + ';' + str(y) + ';' + str(w) + ';' + str(h) + '\n')
        #     cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 2)
        # f.close()

        cv2.imshow('thresh', thresh)
        cv2.imshow('image', image)
        cv2.waitKey()




