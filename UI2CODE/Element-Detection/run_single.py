from os.path import join as pjoin
from turtle import done
import cv2
import os
import glob
import tqdm
import tensorflow as tf

def resize_height_by_longest_edge(img_path, resize_length=800):
    org = cv2.imread(img_path)
    height, width = org.shape[:2]
    if height > width:
        return resize_length
    else:
        return int(resize_length * (height / width))


if __name__ == '__main__':

    '''
        ele:min-grad: gradient threshold to produce binary map         
        ele:ffl-block: fill-flood threshold
        ele:min-ele-area: minimum area for selected elements 
        ele:merge-contained-ele: if True, merge elements contained in others
        text:max-word-inline-gap: words with smaller distance than the gap are counted as a line
        text:max-line-gap: lines with smaller distance than the gap are counted as a paragraph

        Tips:
        1. Larger *min-grad* produces fine-grained binary-map while prone to over-segment element to small pieces
        2. Smaller *min-ele-area* leaves tiny elements while prone to produce noises
        3. If not *merge-contained-ele*, the elements inside others will be recognized, while prone to produce noises
        4. The *max-word-inline-gap* and *max-line-gap* should be dependent on the input image size and resolution

        mobile: {'min-grad':4, 'ffl-block':5, 'min-ele-area':25, 'max-word-inline-gap':6, 'max-line-gap':1}
        web   : {'min-grad':3, 'ffl-block':5, 'min-ele-area':25, 'max-word-inline-gap':4, 'max-line-gap':4}
    '''
    physical_devices = tf.config.experimental.list_physical_devices('CPU')
    # tf.config.experimental.set_memory_growth(0.75)
    # tf.config.experimental.set_memory_growth(physical_devices[0], True)
    # os.environ["CUDA_VISIBLE_DEVICES"]="-1" 
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    # tf.config.set_visible_devices([], 'GPU')

    if tf.test.gpu_device_name():
        print('GPU found')
    else:
        print("No GPU found")

    key_params = {'min-grad':3, 'ffl-block':5, 'min-ele-area':25, 'merge-contained-ele':False,
                  'max-word-inline-gap':4, 'max-line-gap':4}
    with tf.device('/CPU:0'):

        folder_flag = True

        is_ip = True
        is_clf = True
        is_ocr = True
        is_merge = True

        # set input image path
        input_path_img = 'E:\My_MA\FIWItest/testimg/Amazon.png'
        input_folder = 'E:/My_MA/FIWI/Train/data/'
        output_root = 'E:/My_MA/FIWI/after/'

        log_file = output_root + 'error.txt'

        if folder_flag:
            iter_img_list = glob.glob(input_folder + '*.png')
            finished_img_list = os.listdir(output_root + '/mask/')

            print(iter_img_list)

            for input_path_img in tqdm.tqdm(iter_img_list):
                # input_path_img = input_folder + input_path_img
                # print(input_path_img)
                check_name = os.path.splitext(os.path.split(input_path_img)[1])[0] + '_mask.png'
                print(check_name)
                # print(finished_img_list)
                if check_name not in finished_img_list:

                    resized_height = resize_height_by_longest_edge(input_path_img)

                    if is_ocr:
                        import detect_text_east.ocr_east as ocr
                        import detect_text_east.lib_east.eval as eval
                        os.makedirs(pjoin(output_root, 'ocr'), exist_ok=True)
                        models = eval.load()
                        try:
                            ocr.east(input_path_img, output_root, models, key_params['max-word-inline-gap'],
                                resize_by_height=resized_height, show=False)
                        except:
                            with open(log_file, 'a') as f:
                                f.write('\n' + input_path_img)

                    if is_ip:
                        import detect_compo.ip_region_proposal as ip
                        os.makedirs(pjoin(output_root, 'ip'), exist_ok=True)
                        # switch of the classification func
                        classifier = None
                        if is_clf:
                            classifier = {}
                            from cnn.CNN import CNN
                            # classifier['Image'] = CNN('Image')
                            classifier['Elements'] = CNN('Elements')
                            # classifier['Noise'] = CNN('Noise')
                        try:
                            ip.compo_detection(input_path_img, output_root, key_params,
                                        classifier=classifier, resize_by_height=resized_height, show=False)
                        except:
                            with open(log_file, 'a') as f:
                                f.write('\n' + input_path_img)

                    if is_merge:
                        import merge
                        os.makedirs(pjoin(output_root, 'mask'), exist_ok=True)
                        name = os.path.splitext(os.path.split(input_path_img)[1])[0]
                        compo_path = pjoin(output_root, 'ip', str(name) + '.json')
                        ocr_path = pjoin(output_root, 'ocr', str(name) + '.json')
                        save_name = name + '_mask.png'
                        try:
                            merge.incorporate(input_path_img, compo_path, ocr_path, output_root, params=key_params,
                                        resize_by_height=resized_height, show=False, output_name=save_name)
                        except:
                            with open(log_file, 'a') as f:
                                f.write('\n' + name)
                else:
                    print("already Done")
        
        else:
            resized_height = resize_height_by_longest_edge(input_path_img)


            if is_ocr:
                import detect_text_east.ocr_east as ocr
                import detect_text_east.lib_east.eval as eval
                os.makedirs(pjoin(output_root, 'ocr'), exist_ok=True)
                models = eval.load()
                ocr.east(input_path_img, output_root, models, key_params['max-word-inline-gap'],
                        resize_by_height=resized_height, show=False)

            if is_ip:
                import detect_compo.ip_region_proposal as ip
                os.makedirs(pjoin(output_root, 'ip'), exist_ok=True)
                # switch of the classification func
                classifier = None
                if is_clf:
                    classifier = {}
                    from cnn.CNN import CNN
                    # classifier['Image'] = CNN('Image')
                    classifier['Elements'] = CNN('Elements')
                    # classifier['Noise'] = CNN('Noise')
                ip.compo_detection(input_path_img, output_root, key_params,
                                classifier=classifier, resize_by_height=resized_height, show=False)

            if is_merge:
                import merge
                os.makedirs(pjoin(output_root, 'mask'), exist_ok=True)
                name = os.path.splitext(os.path.split(input_path_img)[1])[0]
                compo_path = pjoin(output_root, 'ip', str(name) + '.json')
                ocr_path = pjoin(output_root, 'ocr', str(name) + '.json')
                save_name = name + '_mask.png'
                merge.incorporate(input_path_img, compo_path, ocr_path, output_root, params=key_params,
                                resize_by_height=resized_height, show=False, output_name=save_name)
        