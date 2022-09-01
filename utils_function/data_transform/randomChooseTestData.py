import os, random, shutil
import tqdm

_train_data_path = "E:/My_MA/tmp_data/"
_test_data_path = "E:/My_MA/tmp_val/"
_split_rate = 0.2

def moveFile(fileDir, rate, tarDir):
        pathDir = os.listdir(fileDir + '/data/')    #取图片的原始路径
        labelDir = os.listdir(fileDir + '/label/')
        maskDir = os.listdir(fileDir + '/mask/')
        textMaskDir = os.listdir(fileDir + 'text_mask/')
        print(labelDir)
        filenumber=len(pathDir)
        # picknumber=int(filenumber * _split_rate) #按照rate比例从文件夹中取一定数量图片
        picknumber = 309
        sample = random.sample(pathDir, picknumber)  #随机选取picknumber数量的样本图片
        print (sample)
        for name in tqdm.tqdm(sample):
                print(name)
                shutil.move(fileDir + '/data/' + name, tarDir + '/data/' + name)
                prefix = os.path.splitext(name)[0]
                prefix = prefix.rstrip('_cut')

                # bboxName = prefix + '_bbox.txt'
                heatmapName = prefix + '_heatmap.png'
                puremapName = prefix + '_pureheat.png'
                maskName = prefix + '_cut_mask.png'
                # print(bboxName)
                # shutil.move(fileDir + '/label/' + bboxName, tarDir + '/label/' + bboxName)
                print(heatmapName)
                shutil.move(fileDir + '/label/' + heatmapName, tarDir + '/label/' + heatmapName)
                print(puremapName)
                shutil.move(fileDir + '/label/' + puremapName, tarDir + '/label/' + puremapName)
                print(maskName)
                shutil.move(fileDir + '/mask/' + maskName, tarDir + '/mask/' + maskName)
                shutil.move(fileDir + '/text_mask/' + maskName, tarDir + '/text_mask/' + maskName)
        return

if __name__ == "__main__":
    moveFile(_train_data_path,_split_rate,_test_data_path)