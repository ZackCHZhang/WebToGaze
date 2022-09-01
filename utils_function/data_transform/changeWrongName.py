import os
_path = 'E:/My_MA/GazeMiningDataset/Dataset_stimuli/amazon/shots/gaze-mask/'
for imgName in os.listdir(_path):
    iterName = os.path.splitext(imgName)[0]
    photoNameList = iterName.split('_')
    numberName = photoNameList[len(photoNameList)-2]
    number = numberName.split('.')[0]
    reName = ''
    count = 0
    for iter in photoNameList:
        if count == 2:
            reName = reName + '_' + number
        elif count == 0:
            reName = iter
        else:
            reName = reName + '_' + iter
        count = count + 1
    os.rename(_path + imgName,reName+'.png')