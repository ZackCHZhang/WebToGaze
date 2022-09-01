import os
import cv2
import tqdm
import numpy as np

_save_path = "E:/My_MA/paper_figure/FIWI/"
# _root_path = "E:/My_MA/Results/GazeMiningResult/otherModel/"
_root_path = "E:/My_MA/Results/FIWI/otherModal/"
# _root_path = "E:/My_MA/Results/EyeVidoResult/otherModel/"
# _our_path = "E:/My_MA/Results/Final_res/Gaze/Attention-Line-31/"
_our_path = "E:/My_MA/Results/FIWI/test_res_text1/"
# _label_path = "E:/My_MA/EyevedoDataset/test/label/"
# _label_path = "E:/My_MA/Test/label/"
_label_path = "E:/My_MA/FIWItest/label/"
text_aff_path = _root_path + "middelout_text/"
img_aff_path = _root_path + "middelout_img/"

itList = ["CASD", "DCTS", "HFT", "ICL", "RARE", "SeoMilanfar", "SR"]

for itFile in os.listdir(_our_path):
    if not itFile.endswith(".txt") and not itFile.startswith("A_"):
        imgList = []
        itList = ["CASD", "DCTS", "HFT", "ICL", "RARE", "SeoMilanfar", "SR"]
        for itMethdoe in itList:
            filePath = os.path.join(_root_path, itMethdoe, itFile)
            filePath = filePath.replace(".jpeg", ".png")
            print(filePath)
            imgList.append(cv2.imread(filePath))

        
        img = cv2.imread(os.path.join(_our_path, itFile))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_after = np.zeros(img.shape, dtype=np.uint8)
        h, w = img_after.shape
        tmp = img.mean()
        print(tmp)
        for row in tqdm.tqdm(range(h)):
            for col in range(w):
                it = img[row][col]
                if it >tmp:
                    itc = it-tmp
                    img_after[row][col] = itc
                else:
                    img_after[row][col] = 0
        imgList.append(img_after)
        itList.append("ours")
        label_img = cv2.imread(_label_path + itFile.replace(".jpeg", "_heatmap.png"))
        print(_label_path + itFile.replace(".jpeg", "_heatmap.png"))
        itList.append("GT")
        imgList.append(label_img)
        for i in range(len(imgList)):
            # cv2.imshow("text", imgList[i])
            cv2.imwrite(_save_path + itList[i] + ".png", imgList[i])
            cv2.imshow(itList[i], imgList[i])
        
        cv2.waitKey(10000) == 27

# for itFile in os.listdir(text_aff_path + "aff/"):
#     text_aff_img = cv2.imread( text_aff_path + "aff/" + itFile)
#     text_encoder_img = cv2.imread(text_aff_path + "en/" + itFile)
#     img_aff_img = cv2.imread( img_aff_path + "aff/" + itFile)
#     img_encoder_img = cv2.imread( img_aff_path + "en/" + itFile)
#     result_text = cv2.imread(_root_path + "113/" + itFile)
#     result_img = cv2.imread(_root_path + "114/" + itFile)
#     label_img = cv2.imread(_label_path + itFile.replace(".jpeg", "_heatmap.png"))

#     w = label_img.shape[1]
#     h = label_img.shape[0]
    
#     # text_aff_img = cv2.resize(text_aff_img, (w,h))
#     # text_encoder_img = cv2.resize(text_encoder_img,(w,h))
#     # img_aff_img = cv2.resize(img_aff_img,(w,h))
#     # img_encoder_img = cv2.resize(img_encoder_img,(w,h))
#     # text_encoder_img = cv2.resize(text_encoder_img,  label_img.shape)
#     # text_encoder_img = cv2.resize(text_encoder_img,  label_img.shape)

#     cv2.imshow("textAFF", text_aff_img)
#     cv2.imshow("textEN", text_encoder_img)
#     cv2.imshow("imgAFF", img_aff_img)
#     cv2.imshow("imgEN", img_encoder_img)
#     cv2.imshow("TEXT", result_text)
#     cv2.imshow("IMAGE", result_img)
#     cv2.imshow("label", label_img)
#     cv2.waitKey(60000) == 27