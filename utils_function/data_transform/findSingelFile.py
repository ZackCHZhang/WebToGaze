import os

_datapath = 'E:\My_MA\Training'
_train_path = '/train/'
_label_name = '/label/'

trainList = os.listdir(_datapath + _train_path)
labelList = os.listdir(_datapath + _label_name)

for iter in trainList:
    iterPreffix = iter.rstrip('_cut.png')
    pureHeatName = iterPreffix + '_pureheat.png'
    heatmapName = iterPreffix + '_heatmap.png'
    if pureHeatName not in labelList:
        print(pureHeatName)
    if heatmapName not in labelList:
        print(heatmapName)

for iter in labelList:
    if iter[-7:] == 'cut.png':
        os.remove(_datapath + _label_name + iter)