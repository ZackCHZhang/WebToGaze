import json
import cv2
import glob
import os
from cv2 import imshow
import numpy as np
from yaml import compose

def generate_ocr_mask(datapath, origpath, savepath, ippath):
    data_list = glob.glob(datapath + '*.json')
    for iter in data_list:
        try:
            img_name = os.path.split(iter)[1]

            orig_img_path = origpath + os.path.splitext(img_name)[0] + '.png'
            ip_path = ippath + os.path.splitext(img_name)[0] + '_all.json'
            print(orig_img_path)
            orig_img = cv2.imread(orig_img_path)
            save_img_path = savepath + os.path.splitext(img_name)[0] + '_mask.png'
            
            canves = np.zeros((orig_img.shape[0],orig_img.shape[1]), np.uint8)

            with open(ip_path, 'r') as bk:
                bk_dict = json.loads(bk.read())
                for it in bk_dict['compos']:
                    if it['class'] == "Background":
                        X_radio = orig_img.shape[1]/it["width"]
                        Y_radio = orig_img.shape[0]/it["height"]
                        break

            with open(iter, 'r') as j:
                iter_dict = json.loads(j.read())

                for it in iter_dict['compos']:
                    canves = cv2.rectangle(canves, (int(it['column_min']*Y_radio), int(it['row_min'] * X_radio)), (int(it['column_max'] * Y_radio), int(it['row_max'] * X_radio)), color=(255,255,255), thickness=-1)
        # cv2.imshow('test', canves)
        # cv2.imshow('orig', orig_img)
        # cv2.waitKey(0)

    
            cv2.imwrite(save_img_path,canves)
        except:
            with open(savepath + '/ocrerror.txt', 'a') as f:
                f.write('\n' + img_name)



if __name__ == '__main__':

    name = "E:/My_MA/FIWI/Train/data/takungpao.png"
    img = cv2.imread(name)
    canves = np.zeros((img.shape[0],img.shape[1]), np.uint8)
    cv2.imwrite("E:/My_MA/FIWItest/takungpao.png",canves)


    # _data_path = "E:/My_MA/FIWI/after/ocr/"
    # _orig_path = "E:/My_MA/FIWI/Train/data/"
    # _save_path = "E:/My_MA/FIWI/after/text_mask/"
    # _ip_path = "E:/My_MA/FIWI/after/ip/"
    # generate_ocr_mask(_data_path, _orig_path, _save_path,_ip_path)