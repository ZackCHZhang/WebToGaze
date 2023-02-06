# WebToGaze
This work proposes visual saliency detection, which predicts the attention of humans to a given image. By performing proper cross-task model transfers, the automatic generation of the user's gaze (as heatmaps, for instance) from a website screenshot (as an input substitute) is realized.
Specifically, a state-of-the-art general saliency detection model is chosen as the baseline approach, which is then fine-tuned to work (or specialize) with website screenshot inputs. 
Furthermore, based on the original modal a new model was proposed, which applied the visual information from a website, i.e. the image mask and text mask of a website were utilized in the model. 

# Data Structure
In Utils_function is some python files which aims to generate and prepocessing dataset from MySql dataset. In addtion also some file to achive the metrrics funxtions. Besides, some generatte figure methode also inside this folder.

In saliency folder is the models, which achive the P-MSI, P-MSI-S-A, etc. models.

In UI2CODE folder is the code based on MUlong et. al. function and with  changed to implement the Mask generation function in this work.

In Other_Model folders is the models from other researchers, applied here aims to make the state-of-art in this work.

# Dataset Download
Because the size of dataset size:
GazeMining Orignal Dataset Link: https://drive.google.com/file/d/1wCg493QVQU_eGVlgPJH_mRgEppE-skUk/view?usp=sharing
Contrasitive Websites Dataset Link: https://drive.google.com/file/d/1QawwMYB9Vva6klUhgbXqmWBUhENeSdpd/view?usp=sharing
FiWI after prepossing Dataset link: https://drive.google.com/file/d/1pgSF27-U1ZoVgWuHzzOTDeD0t52Y2daW/view?usp=sharing
Combined Dataset Link: https://drive.google.com/file/d/1NQLSyk_Ldgrvv2OOPBaAFlxASAaDH3tL/view?usp=sharing

Also State-of-Arts models and weights are also stored in google drive: https://drive.google.com/file/d/1Nl27PU1HxdOoT54s6pykxteLEvLMGQ8c/view?usp=sharing
The work can be find here: https://drive.google.com/file/d/1UUhFHbgQF5Qa0-8ZHzgx0IFZUIpfLr44/view?usp=sharing

This work was accetped by VISAPP 2023

