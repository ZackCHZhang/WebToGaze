import os
from unicodedata import name
import cv2
import numpy as np
import os
 
_data_path = "E:/My_MA/paper_figure/FIWI/"
_save_path = "E:/My_MA/paper_figure/FIWI/res/"

def heat_map(dataPath, savePath, name):
    fix = cv2.imread(dataPath + name)
    max_val = np.max(fix)
    min_val = np.min(fix)
    fix = (fix - min_val)/(max_val - min_val)
    fix *= 255
    pred_ = fix.astype(np.uint8)
    pred_heat_map = cv2.applyColorMap(pred_, cv2.COLORMAP_JET)
    cv2.imwrite(savePath + "HFT2*_h.jpg", pred_heat_map)
 
def heat_map_folder(folder_path, savepath):
    for it in os.listdir(folder_path):
         if it.endswith(".png"):
            fix = cv2.imread(folder_path + it)
            max_val = np.max(fix)
            min_val = np.min(fix)
            fix = (fix - min_val)/(max_val - min_val)
            fix *= 255
            pred_ = fix.astype(np.uint8)
            pred_heat_map = cv2.applyColorMap(pred_, cv2.COLORMAP_JET)
            cv2.imwrite(savepath + it, pred_heat_map) 

if __name__ == "__main__":
    # pathDir = os.listdir(_data_path)
    # for name in pathDir:
    #     heat_map(_data_path,_save_path, name)
    # heat_map(_data_path,_save_path,"HFT2.png")
    heat_map_folder(_data_path, _save_path)
    
    print("\n done!\n")