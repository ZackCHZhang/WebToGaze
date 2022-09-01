import cv2
import numpy as np
import os

_gc_folder = "E:/My_MA/Results/paper_rebuild/eyevedo/test_gc_without/"
# _label_path = "E:/My_MA/Test/testimg/"
_label_path = "E:/My_MA/EyevedoDataset/test/data/"

for it in os.listdir(_gc_folder):
    if not it.startswith("A") and it != "after":
        img = cv2.imread(_gc_folder+it)
        # img = 255 - img
        # img = (img - np.min(img))/(np.max(img)-np.min(img))
        # img *= 255
        heatmap = cv2.applyColorMap(img.astype(np.uint8),cv2.COLORMAP_JET)
        oriImg = cv2.imread(_label_path + it.replace(".jpeg",".jpg"))
        heatmap = cv2.resize(heatmap, (oriImg.shape[1],oriImg.shape[0]))
        mergImg = cv2.addWeighted(oriImg,0.5,heatmap,0.5,0)
        mergImg = np.minimum(mergImg, 255.0).astype(np.uint8)
        mergImg = cv2.cvtColor(mergImg,cv2.COLOR_BGR2RGB)

        cv2.imwrite(_gc_folder + "/after/" + it, mergImg)
        # cv2.imshow("t",mergImg)
        # cv2.waitKey(0)