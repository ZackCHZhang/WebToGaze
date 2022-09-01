import os
import glob

def del_all_file(path,name):
    rmList = glob.glob(path + '/' + name + '*')
    for file in rmList:
        print(file)
        os.remove(file)

def del_error_img(datapath,labelpath,maskpath,textmaskpath,logpath):
    fileList = []
    with open(logpath, 'r') as log:
        for line in log:

            fileList.append(line.split('\n')[0])

    print(fileList)
    for itFileName in fileList:
        if itFileName.endswith('.json'):
            delName = os.path.splitext(itFileName)[0].replace('_cut','_')
        else:
            delName = itFileName.replace('_cut','_')
        print(delName)
        del_all_file(datapath,delName)
        del_all_file(labelpath,delName)
        del_all_file(maskpath,delName)
        del_all_file(textmaskpath,delName)

if __name__ == '__main__':
    _datapath = 'E:/My_MA/Val/train/'
    _labelpath = 'E:/My_MA/Val/label/'
    _maskpath = 'E:/My_MA/Val/mask/'
    _text_mask_path = 'E:/My_MA/Val/text_mask/'
    _error_log_path = 'E:/My_MA/Val/error.txt'

    del_error_img(_datapath,_labelpath,_maskpath,_text_mask_path,_error_log_path)
    