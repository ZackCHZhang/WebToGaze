import cv2
from cv2 import imread
import numpy as np
import tqdm

def post_processing_img(path, name):
    img = imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_after = np.zeros(img.shape, dtype=np.uint8)
    # img_after = cv2.equalizeHist(img)
    # img_after = cv2.GaussianBlur(img, (5,5),0)
    h, w = img_after.shape
    
    # 遍历像素点，修改图像b,g,r值
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
    cv2.imwrite("E:/My_MA/Results/paper_rebuild/eyevedo/test_gc_without/after/"+ name + ".png", img_after)
    cv2.imshow("before", img)
    cv2.imshow("after", img_after)
    cv2.waitKey(0)


if __name__ == "__main__":
    test_img_path = "E:/My_MA/Results/paper_rebuild/eyevedo/test_gc_without/after/"
    name = "F_U12_S2-1903394518_1301903575_orig_pred"
    path = test_img_path + name + ".jpeg"
    post_processing_img(path,name)