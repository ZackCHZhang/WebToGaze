# from asyncio import FastChildWatcher
# from distutils.log import error
# from operator import index
import os
# from re import T, X
from sys import path
from unicodedata import name
from matplotlib import image
from matplotlib.pyplot import axes, axis

import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.tools.graph_transforms import TransformGraph
from tensorflow.python.tools import inspect_checkpoint as chkp
# import tensorflow_addons as tfa
# from tensorflow import ins

import config
import download
import loss




class AFF:
    def __init__(self, n_channel = 512, r = 4,data_format="channels_first", basename = "AFF"):
        # self.gamma = gamma
        self.output = None
        self.inter_channel = n_channel//r 
        self.n_channel = n_channel
        self._data_format = data_format
        self._basename = basename + "/"

    def _local_att1(self, input):

        layer01 = tf.layers.conv2d(input, self.inter_channel, 1,
                    # padding="same",
                    activation= None,
                    data_format=self._data_format,
                    # trainable=trainFlag,
                    name= self._basename + "la1_1",
                    reuse=tf.AUTO_REUSE)
        layer02 = tf.layers.BatchNormalization(axis=1, name = self._basename + "laBN1_1")(layer01,)
        layer03 = tf.nn.relu(layer02)
        layer04 = tf.layers.conv2d(layer03, self.n_channel, 1,
            # padding="same",
            activation= None,
            data_format=self._data_format,
            # trainable=trainFlag,
            name=self._basename + "la1_2",
            reuse=tf.AUTO_REUSE)
        return tf.layers.BatchNormalization(axis=1, name = self._basename + "laBN1_2")(layer04)
    


    def _global_att1(self, input):
        layer01 = tf.keras.layers.GlobalAveragePooling2D(data_format = self._data_format)(input)
        # layer01 = tf.layers.average_pooling2d(
        #     input, 1,
        #     # padding='valid', 
        #     data_format=self._data_format,
        #     # name= "IAFF/ga1"
        # )
        layer01 = tf.expand_dims(layer01, axis =-1)
        layer01 = tf.expand_dims(layer01, axis =-1)
        layer02 = tf.layers.conv2d(layer01, self.inter_channel, 1,
            # padding="same",
            activation= None,
            data_format=self._data_format,
            # trainable=trainFlag,
            name=self._basename + "ga1_1",
            reuse=tf.AUTO_REUSE)
        layer03 = tf.layers.BatchNormalization(axis=1, name = self._basename + "gaBN1_1")(layer02)
        layer04 = tf.nn.relu(layer03)
        layer05 = tf.layers.conv2d(layer04, self.n_channel, 1,
            # padding="same",
            activation= None,
            data_format=self._data_format,
            # trainable=trainFlag,
            name=self._basename + "ga1_2",
            reuse=tf.AUTO_REUSE)
        return tf.layers.BatchNormalization(axis=1, name = self._basename + "gaBN1_2")(layer05)
         
    def forward(self, x , residual):
        xa = tf.add(x, residual)
        xl = self._local_att1(xa)
        xg = self._global_att1(xa)
        xlg = tf.add(xl,xg)
        wei = tf.nn.sigmoid(xlg)
        xi = x * wei + residual * (1-wei)

        xo = 2 * x * wei + 2 * residual * (1-wei)
        return xo


class IAFF:
    def __init__(self, n_channel = 512, r = 4,data_format="channels_first", basename = "IAFF"):
        # self.gamma = gamma
        self.output = None
        self.inter_channel = n_channel//r 
        self.n_channel = n_channel
        self._data_format = data_format
        self._basename = basename + "/"

    def _local_att1(self, input):

        layer01 = tf.layers.conv2d(input, self.inter_channel, 1,
                    # padding="same",
                    activation= None,
                    data_format=self._data_format,
                    # trainable=trainFlag,
                    name= self._basename + "la1_1",
                    reuse=tf.AUTO_REUSE)
        layer02 = tf.layers.BatchNormalization(axis=1)(layer01,)
        layer03 = tf.nn.relu(layer02)
        layer04 = tf.layers.conv2d(layer03, self.n_channel, 1,
            # padding="same",
            activation= None,
            data_format=self._data_format,
            # trainable=trainFlag,
            name=self._basename + "la1_2",
            reuse=tf.AUTO_REUSE)
        return tf.layers.BatchNormalization(axis=1)(layer04)
    

    def _local_att2(self, input):
        layer01 = tf.layers.conv2d(input, self.inter_channel, 1,
                    # padding="same",
                    activation= None,
                    data_format=self._data_format,
                    # trainable=trainFlag,
                    name=self._basename + "la2_1",
                    reuse=tf.AUTO_REUSE)
        layer02 = tf.layers.BatchNormalization(axis=1)(layer01)
        layer03 = tf.nn.relu(layer02)
        layer04 = tf.layers.conv2d(layer03, self.n_channel, 1,
            # padding="same",
            activation= None,
            data_format=self._data_format,
            # trainable=trainFlag,
            name=self._basename + "la2_2",
            reuse=tf.AUTO_REUSE)
        return tf.layers.BatchNormalization(axis=1)(layer04)     


    def _global_att1(self, input):
        layer01 = tf.keras.layers.GlobalAveragePooling2D(data_format = self._data_format)(input)
        # layer01 = tf.layers.average_pooling2d(
        #     input, 1,
        #     # padding='valid', 
        #     data_format=self._data_format,
        #     # name= "IAFF/ga1"
        # )
        layer01 = tf.expand_dims(layer01, axis =-1)
        layer01 = tf.expand_dims(layer01, axis =-1)
        layer02 = tf.layers.conv2d(layer01, self.inter_channel, 1,
            # padding="same",
            activation= None,
            data_format=self._data_format,
            # trainable=trainFlag,
            name=self._basename + "ga1_1",
            reuse=tf.AUTO_REUSE)
        layer03 = tf.layers.BatchNormalization(axis=1)(layer02)
        layer04 = tf.nn.relu(layer03)
        layer05 = tf.layers.conv2d(layer04, self.n_channel, 1,
            # padding="same",
            activation= None,
            data_format=self._data_format,
            # trainable=trainFlag,
            name=self._basename + "ga1_2",
            reuse=tf.AUTO_REUSE)
        return tf.layers.BatchNormalization(axis=1)(layer05)
         
    def _global_att2(self, input):
        layer01 = tf.keras.layers.GlobalAveragePooling2D(data_format = self._data_format)(input)
        # layer01 = tf.layers.average_pooling2d(
        #     input, 1, 1,
        #     # padding='valid', 
        #     data_format=self._data_format,
        #     # name= "IAFF/ga1"
        # )
        layer01 = tf.expand_dims(layer01, axis =-1)
        layer01 = tf.expand_dims(layer01, axis =-1)
        layer02 = tf.layers.conv2d(layer01, self.inter_channel, 1,
            # padding="same",
            activation= None,
            data_format=self._data_format,
            # trainable=trainFlag,
            name=self._basename + "ga2_1",
            reuse=tf.AUTO_REUSE)
        layer03 = tf.layers.BatchNormalization(axis=1)(layer02)
        layer04 = tf.nn.relu(layer03)
        layer05 = tf.layers.conv2d(layer04, self.n_channel, 1,
            # padding="same",
            activation= None,
            data_format=self._data_format,
            # trainable=trainFlag,
            name=self._basename + "ga2_2",
            reuse=tf.AUTO_REUSE)
        return tf.layers.BatchNormalization(axis=1)(layer05)
    
    def forward(self, x , residual):
        xa = tf.add(x, residual)
        xl = self._local_att1(xa)
        xg = self._global_att1(xa)
        xlg = tf.add(xl,xg)
        wei = tf.nn.sigmoid(xlg)
        xi = x * wei + residual * (1-wei)

        xl2 = self._local_att2(xi)
        xg2 = self._global_att2(xi)
        xlg2 = tf.add(xl2,xg2)
        wei2 = tf.nn.sigmoid(xlg2)
        return x * wei2 + residual * (1-wei2)

         
if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = "3"
    # device = torch.device("cuda:0")

    x, residual= tf.ones([8,64, 32, 32]),tf.ones([8,64, 32, 32])
    channels=x.shape[1]

    model=AFF(n_channel=channels)
    one_op = model.forward(x, residual)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        res = sess.run(one_op)
        print(res)
    # output = model(x, residual)
    # print(output.shape)        



        

class DISCRIMINATOR:
    def __init__(self,MMFlag=False,DMFlag=False,shareflag=False):
        self._discriminator_real_output = None
        self._discriminator_fake_output = None
        self._loss_function = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        if config.PARAMS["device"] == "gpu":
            self._data_format = "channels_first"
            self._channel_axis = 1
            self._dims_axis = (2, 3)
        elif config.PARAMS["device"] == "cpu":
            self._data_format = "channels_last"
            self._channel_axis = 3
            self._dims_axis = (1, 2)

    def _discriminator(self, image):
        trainFlag = True
        print(image.get_shape())
        image = tf.image.resize(image,config.DIMS["image_size_salicon"])
        print(image.get_shape())
        if self._data_format == "channels_first":
            image = tf.transpose(image, (0, 3, 1, 2))
        
        with tf.variable_scope('Disc'):
            layer01 = tf.layers.conv2d(image, 3, 1,
                                padding="same",
                                activation=tf.nn.relu,
                                data_format=self._data_format,
                                trainable=trainFlag,
                                name="disc/conv1_1",
                                reuse=tf.AUTO_REUSE)

            layer02 = tf.layers.conv2d(layer01, 32, 3,
                                padding="same",
                                activation=tf.nn.relu,
                                data_format=self._data_format,
                                trainable=trainFlag,
                                name="disc/conv1_2",
                                reuse=tf.AUTO_REUSE)
            layer06 = tf.layers.max_pooling2d(layer02, 2, 2,
                data_format=self._data_format)

            layer07 = tf.layers.conv2d(layer06, 64, 3,
                                padding="same",
                                activation=tf.nn.relu,
                                data_format=self._data_format,
                                trainable=trainFlag,
                                name="disc/conv2_1",
                                reuse=tf.AUTO_REUSE)

            layer08 = tf.layers.conv2d(layer07, 64, 3,
                                padding="same",
                                activation=tf.nn.relu,
                                data_format=self._data_format,
                                trainable=trainFlag,
                                name="disc/conv2_2",
                                reuse=tf.AUTO_REUSE)
            layer09 = tf.layers.max_pooling2d(layer08, 2, 2,
                data_format=self._data_format)

            layer10 = tf.layers.conv2d(layer09, 64, 3,
                                padding="same",
                                activation=tf.nn.relu,
                                data_format=self._data_format,
                                trainable=trainFlag,
                                name="disc/conv3_1",
                                reuse=tf.AUTO_REUSE)

            layer11 = tf.layers.conv2d(layer10, 64, 3,
                                padding="same",
                                activation=tf.nn.relu,
                                data_format=self._data_format,
                                trainable=trainFlag,
                                name="disc/conv3_2",
                                reuse=tf.AUTO_REUSE)
            layer12 = tf.layers.max_pooling2d(layer11, 2, 2,
                data_format=self._data_format)

            # dim = tf.reduce_prod(tf.shape(layer12)[1:])
            # faltten = tf.keras.layers.Flatten(data_format=self._data_format)(layer12)
            # faltten = tf.layers.
            faltten = tf.reshape(layer12,[8,76800])
            layer13 = tf.layers.dense(faltten, 100,
                                activation=tf.nn.tanh,
                                trainable=trainFlag,
                                name="disc/fc1",reuse=tf.AUTO_REUSE)

            layer14 = tf.layers.dense(layer13,2,
                                activation=tf.nn.tanh,
                                trainable=trainFlag,
                                name="disc/fc2",reuse=tf.AUTO_REUSE)
            layer15 = tf.layers.dense(layer14,1,
                                activation=tf.nn.sigmoid,
                                trainable=trainFlag,
                                name="disc/fc3",reuse=tf.AUTO_REUSE)
            return layer15
    
    def forward(self, input):
        self._discriminator_real_output = self._discriminator(input)
        # self._discriminator_fake_output = self._discriminator(fake_input)

        return self._discriminator_real_output
    def train(self, real_output, fake_output,learning_rate):

        real_error = self._loss_function(tf.ones_like(real_output), real_output)
        fake_error = self._loss_function(tf.zeros_like(fake_output), fake_output)
        error = real_error + fake_error
        optimizer = tf.train.AdamOptimizer(learning_rate)


        return optimizer, error

    def save(self, saver, sess, dataset, path, device, newMMFlag = False, DMFlag = False, epoch = ""):
        """This saves a model checkpoint to disk and creates
           the folder if it doesn't exist yet.

        Args:
            saver (object): An object for saving the model.
            sess (object): The current TF training session.
            path (str): The path used for saving the model.
            device (str): Represents either "cpu" or "gpu".
        """

        os.makedirs(path, exist_ok=True)
        if newMMFlag:
            saver.save(sess, path + "model_disc_%s_%s_newMM%s.ckpt" % (dataset, device, epoch),
                   write_meta_graph=False, write_state=False)
        elif DMFlag:
            saver.save(sess, path + "model_disc_%s_%s_newDM%s.ckpt" % (dataset, device, epoch),
                   write_meta_graph=False, write_state=False)
        else:
            saver.save(sess, path + "model_disc_%s_%s.ckpt%s" % (dataset, device, epoch),
                   write_meta_graph=False, write_state=False)

    def restore(self, sess, dataset, paths, device):
        """This function allows continued training from a prior checkpoint and
           training from scratch with the pretrained VGG16 weights. In case the
           dataset is either CAT2000 or MIT1003, a prior checkpoint based on
           the SALICON dataset is required.

        Args:
            sess (object): The current TF training session.
            dataset ([type]): The dataset used for training.
            paths (dict, str): A dictionary with all path elements.
            device (str): Represents either "cpu" or "gpu".

        Returns:
            object: A saver object for saving the model.
        """
        VGG16Flag = True

        model_name = "model_disc_%s_%s" % (dataset, device)
        salicon_name = "model_disc_salicon_%s" % device
        vgg16_name = "vgg16_hybrid"
        # fintuning之后继续训练添加mask的
        eyevedo_name = "model_disc_alldataset_%s" %device

        ext1 = ".ckpt.data-00000-of-00001"
        ext2 = ".ckpt.index"

        saver = tf.train.Saver()
            
        if os.path.isfile(paths["latest"] + model_name + ext1) and \
           os.path.isfile(paths["latest"] + model_name + ext2):
            print(paths["latest"] + model_name + ".ckpt")
            saver.restore(sess, paths["latest"] + model_name + ".ckpt")
        # 添加判断fintuning之后训练
        elif os.path.isfile(paths["latest"] + eyevedo_name + ext1) and \
           os.path.isfile(paths["latest"] + eyevedo_name + ext2):
            # chkp.print_tensors_in_checkpoint_file(paths["latest"] + eyevedo_name + ".ckpt", tensor_name=None, all_tensors = False)
            # variables = tf.contrib.framework.get_variables_to_restore()
            # # 加载finetuning模型 去除前两个参数
            # vars_to_restore = [v for v in variables]
            # vars_to_restore.pop(0)
            # vars_to_restore.pop(0)
            # loader = tf.train.Saver(vars_to_restore)
            # loader.restore(sess, paths["latest"] + eyevedo_name + ".ckpt")
            # self._pretraining()
            # loader = tf.train.Saver(self._mapping)
            # loader.restore(sess, paths["latest"] + eyevedo_name + ".ckpt")
            saver.restore(sess, paths["latest"] + eyevedo_name + ".ckpt")
            

        # else:
        #     if not (os.path.isfile(paths["weights"] + vgg16_name + ext1) or
        #             os.path.isfile(paths["weights"] + vgg16_name + ext2)):
        #         download.download_pretrained_weights(paths["weights"],
        #                                              "vgg16_hybrid")
        #     self._pretraining()

        #     loader = tf.train.Saver(self._mapping)
        #     loader.restore(sess, paths["weights"] + vgg16_name + ".ckpt")

        return saver

    def optimize(self, sess, dataset, path, device, newMMFlag=False, DMFlag=False,epoch = ""):
        """The best performing model is frozen, optimized for inference
           by removing unneeded training operations, and written to disk.

        Args:
            sess (object): The current TF training session.
            path (str): The path used for saving the model.
            device (str): Represents either "cpu" or "gpu".

        .. seealso:: https://bit.ly/2VBBdqQ and https://bit.ly/2W7YqBa
        """
        if newMMFlag:
            model_name = "model_disc_%s_%s_newMM%s" % (dataset, device,epoch)
        elif DMFlag:
            model_name = "model_disc_%s_%s_newDM%s" % (dataset, device,epoch)
        else:
            model_name = "model_disc_%s_%s%s" % (dataset, device,epoch)
        model_path = path + model_name

        tf.train.write_graph(sess.graph.as_graph_def(),
                             path, model_name + ".pbtxt")

        freeze_graph.freeze_graph(model_path + ".pbtxt", "", False,
                                  model_path + ".ckpt", "output",
                                  "save/restore_all", "save/Const:0",
                                  model_path + ".pb", True, "")

        os.remove(model_path + ".pbtxt")

        graph_def = tf.GraphDef()

        with tf.gfile.Open(model_path + ".pb", "rb") as file:
            graph_def.ParseFromString(file.read())

        transforms = ["remove_nodes(op=Identity)",
                      "merge_duplicate_nodes",
                      "strip_unused_nodes",
                      "fold_constants(ignore_errors=true)"]

        optimized_graph_def = TransformGraph(graph_def,
                                             ["input"],
                                             ["output"],
                                             transforms)

        tf.train.write_graph(optimized_graph_def,
                             logdir=path,
                             as_text=False,
                             name=model_name + ".pb")




class MSINET:
    """The class representing the MSI-Net based on the VGG16 model. It
       implements a definition of the computational graph, as well as
       functions related to network training.
    """

    def __init__(self,MMFlag=False,DMFlag=False,shareflag=False, attentionFlag = False):
        self._output = None
        self._maskoutput = None
        self._secondmaskoutput = None
        self._discriminator_output = None
        self.MMFlag = MMFlag
        self.DMFlag = DMFlag
        self.shareFlag = shareflag
        self._mapping = {}
        self._atflag = attentionFlag
        if attentionFlag:
            self._atflag = True
            self._iAFFas = AFF(n_channel=512, basename= "AFFas")
            self._iAFFen = AFF(n_channel=512, basename="AFFen")
            self._iAFF06 = AFF(n_channel=128, basename= "AFF06")
            self._iAFF10 = AFF(n_channel=256, basename="AFF10")
            self._iAFF18 = AFF(n_channel=512, basename="AFF18")

        if config.PARAMS["device"] == "gpu":
            self._data_format = "channels_first"
            self._channel_axis = 1
            self._dims_axis = (2, 3)
        elif config.PARAMS["device"] == "cpu":
            self._data_format = "channels_last"
            self._channel_axis = 3
            self._dims_axis = (1, 2)

    def _secondmaskencoder(self, mask):
        trainable = True
        if self._data_format == "channels_first":
            mask = tf.transpose(mask, (0, 3, 1, 2))

        layer01 = tf.layers.conv2d(mask, 64, 3,
                                   padding="same",
                                   activation=tf.nn.relu,
                                   data_format=self._data_format,
                                   trainable=trainable,
                                   name="secmaskconv/conv1_1")

        layer02 = tf.layers.conv2d(layer01, 64, 3,
                                   padding="same",
                                   activation=tf.nn.relu,
                                   data_format=self._data_format,
                                   trainable=trainable,
                                   name="secmaskconv/conv1_2")
        # print(layer02.shape)
        layer03 = tf.layers.max_pooling2d(layer02, 2, 2,
                                          data_format=self._data_format)

        layer04 = tf.layers.conv2d(layer03, 128, 3,
                                   padding="same",
                                   activation=tf.nn.relu,
                                   data_format=self._data_format,
                                   trainable=trainable,
                                   name="secmaskconv/conv2_1")

        layer05 = tf.layers.conv2d(layer04, 128, 3,
                                   padding="same",
                                   activation=tf.nn.relu,
                                   data_format=self._data_format,
                                   trainable=trainable,
                                   name="secmaskconv/conv2_2")

        layer06 = tf.layers.max_pooling2d(layer05, 2, 2,
                                          data_format=self._data_format)

        layer07 = tf.layers.conv2d(layer06, 256, 3,
                                   padding="same",
                                   activation=tf.nn.relu,
                                   data_format=self._data_format,
                                   trainable=trainable,
                                   name="secmaskconv/conv3_1")

        layer08 = tf.layers.conv2d(layer07, 256, 3,
                                   padding="same",
                                   activation=tf.nn.relu,
                                   data_format=self._data_format,
                                   trainable=trainable,
                                   name="secmaskconv/conv3_2")

        layer09 = tf.layers.conv2d(layer08, 256, 3,
                                   padding="same",
                                   activation=tf.nn.relu,
                                   data_format=self._data_format,
                                   trainable=trainable,
                                   name="secmaskconv/conv3_3")

        layer10 = tf.layers.max_pooling2d(layer09, 2, 2,
                                          data_format=self._data_format)

        layer11 = tf.layers.conv2d(layer10, 512, 3,
                                   padding="same",
                                   activation=tf.nn.relu,
                                   data_format=self._data_format,
                                   trainable=trainable,
                                   name="secmaskconv/conv4_1")
        
        self._secondmaskoutput = layer11

    def _maskencoder(self, mask):
        trainable = True

        if self._data_format == "channels_first":
            mask = tf.transpose(mask, (0, 3, 1, 2))


        layer01 = tf.layers.conv2d(mask, 64, 3,
                                padding="same",
                                activation=tf.nn.relu,
                                data_format=self._data_format,
                                trainable=trainable,
                                name="maskconv/conv1_1")

        layer02 = tf.layers.conv2d(layer01, 64, 3,
                                padding="same",
                                activation=tf.nn.relu,
                                data_format=self._data_format,
                                trainable=trainable,
                                name="maskconv/conv1_2")
        # print(layer02.shape)
        layer03 = tf.layers.max_pooling2d(layer02, 2, 2,
                                        data_format=self._data_format)

        layer04 = tf.layers.conv2d(layer03, 128, 3,
                                padding="same",
                                activation=tf.nn.relu,
                                data_format=self._data_format,
                                trainable=trainable,
                                name="maskconv/conv2_1")

        layer05 = tf.layers.conv2d(layer04, 128, 3,
                                padding="same",
                                activation=tf.nn.relu,
                                data_format=self._data_format,
                                trainable=trainable,
                                name="maskconv/conv2_2")

        layer06 = tf.layers.max_pooling2d(layer05, 2, 2,
                                        data_format=self._data_format)

        layer07 = tf.layers.conv2d(layer06, 256, 3,
                                padding="same",
                                activation=tf.nn.relu,
                                data_format=self._data_format,
                                trainable=trainable,
                                name="maskconv/conv3_1")

        layer08 = tf.layers.conv2d(layer07, 256, 3,
                                padding="same",
                                activation=tf.nn.relu,
                                data_format=self._data_format,
                                trainable=trainable,
                                name="maskconv/conv3_2")

        layer09 = tf.layers.conv2d(layer08, 256, 3,
                                padding="same",
                                activation=tf.nn.relu,
                                data_format=self._data_format,
                                trainable=trainable,
                                name="maskconv/conv3_3")

        layer10 = tf.layers.max_pooling2d(layer09, 2, 2,
                                        data_format=self._data_format)

        layer11 = tf.layers.conv2d(layer10, 512, 3,
                                padding="same",
                                activation=tf.nn.relu,
                                data_format=self._data_format,
                                trainable=trainable,
                                name="maskconv/conv4_1")
        
        layer11 = tf.layers.BatchNormalization(axis=1, name="maskBN")(layer11)

        # layer11 = tf.layers.max_pooling2d(layer11, 2, 2,
        #                         data_format=self._data_format)
        
        self._maskoutput = layer11

        return layer06, layer10, layer11




    def _encoder(self, images, mask_flag=False):
        """The encoder of the model consists of a pretrained VGG16 architecture
           with 13 convolutional layers. All dense layers are discarded and the
           last 3 layers are dilated at a rate of 2 to account for the omitted
           downsampling. Finally, the activations from 3 layers are combined.

        Args:
            images (tensor, float32): A 4D tensor that holds the RGB image
                                      batches used as input to the network.
        """

        trainFlag = True
        if mask_flag:
            print(images.shape)
            # double mask if change one mask line47 [1,1,1,4] aslo delete one 0 in line 46
            imagenet_mean = tf.constant([103.939, 116.779, 123.68, 0, 0])
            imagenet_mean = tf.reshape(imagenet_mean, [1, 1, 1, 5])
        else:
            imagenet_mean = tf.constant([103.939, 116.779, 123.68])
            imagenet_mean = tf.reshape(imagenet_mean, [1, 1, 1, 3])

        images -= imagenet_mean

        with tf.variable_scope('Gen'):

            if self._data_format == "channels_first":
                images = tf.transpose(images, (0, 3, 1, 2))

            layer01 = tf.layers.conv2d(images, 64, 3,
                                    padding="same",
                                    activation=tf.nn.relu,
                                    data_format=self._data_format,
                                    trainable=trainFlag,
                                    name="conv1/conv1_1")

            layer02 = tf.layers.conv2d(layer01, 64, 3,
                                    padding="same",
                                    activation=tf.nn.relu,
                                    data_format=self._data_format,
                                    trainable=trainFlag,
                                    name="conv1/conv1_2")
            # print(layer02.shape)
            layer03 = tf.layers.max_pooling2d(layer02, 2, 2,
                                            data_format=self._data_format)

            layer04 = tf.layers.conv2d(layer03, 128, 3,
                                    padding="same",
                                    activation=tf.nn.relu,
                                    data_format=self._data_format,
                                    trainable=trainFlag,
                                    name="conv2/conv2_1")

            layer05 = tf.layers.conv2d(layer04, 128, 3,
                                    padding="same",
                                    activation=tf.nn.relu,
                                    data_format=self._data_format,
                                    trainable=trainFlag,
                                    name="conv2/conv2_2")

            layer06 = tf.layers.max_pooling2d(layer05, 2, 2,
                                            data_format=self._data_format)

            layer07 = tf.layers.conv2d(layer06, 256, 3,
                                    padding="same",
                                    activation=tf.nn.relu,
                                    data_format=self._data_format,
                                    trainable=trainFlag,
                                    name="conv3/conv3_1")

            layer08 = tf.layers.conv2d(layer07, 256, 3,
                                    padding="same",
                                    activation=tf.nn.relu,
                                    data_format=self._data_format,
                                    trainable=trainFlag,
                                    name="conv3/conv3_2")

            layer09 = tf.layers.conv2d(layer08, 256, 3,
                                    padding="same",
                                    activation=tf.nn.relu,
                                    data_format=self._data_format,
                                    trainable=trainFlag,
                                    name="conv3/conv3_3")

            layer10 = tf.layers.max_pooling2d(layer09, 2, 2,
                                            data_format=self._data_format)

            layer11 = tf.layers.conv2d(layer10, 512, 3,
                                    padding="same",
                                    activation=tf.nn.relu,
                                    data_format=self._data_format,
                                    trainable=trainFlag,
                                    name="conv4/conv4_1")

            layer12 = tf.layers.conv2d(layer11, 512, 3,
                                    padding="same",
                                    activation=tf.nn.relu,
                                    data_format=self._data_format,
                                    trainable=trainFlag,
                                    name="conv4/conv4_2")

            layer13 = tf.layers.conv2d(layer12, 512, 3,
                                    padding="same",
                                    activation=tf.nn.relu,
                                    data_format=self._data_format,
                                    trainable=trainFlag,
                                    name="conv4/conv4_3")

            layer14 = tf.layers.max_pooling2d(layer13, 2, 1,
                                            padding="same",
                                            data_format=self._data_format)

            layer15 = tf.layers.conv2d(layer14, 512, 3,
                                    padding="same",
                                    activation=tf.nn.relu,
                                    dilation_rate=2,
                                    data_format=self._data_format,
                                    trainable=trainFlag,
                                    name="conv5/conv5_1")

            layer16 = tf.layers.conv2d(layer15, 512, 3,
                                    padding="same",
                                    activation=tf.nn.relu,
                                    dilation_rate=2,
                                    data_format=self._data_format,
                                    trainable=trainFlag,
                                    name="conv5/conv5_2")

            layer17 = tf.layers.conv2d(layer16, 512, 3,
                                    padding="same",
                                    activation=tf.nn.relu,
                                    dilation_rate=2,
                                    data_format=self._data_format,
                                    trainable=trainFlag,
                                    name="conv5/conv5_3")

            layer18 = tf.layers.max_pooling2d(layer17, 2, 1,
                                            padding="same",
                                            data_format=self._data_format)
            
            if self._atflag:
                # encoder_output = self._iAFF.forward(layer10,layer14)
                encoder_output = self._iAFFen.forward(layer14,layer18)
                # encoder_output = tf.add(encoder_output, layer10)
            else:

                encoder_output = tf.concat([layer10, layer14, layer18],
                                    axis=self._channel_axis)

            self._output = encoder_output

            return layer06, layer10, layer18

    def _aspp(self, features, mask=None, secmask=None):
        """The ASPP module samples information at multiple spatial scales in
           parallel via convolutional layers with different dilation factors.
           The activations are then combined with global scene context and
           represented as a common tensor.

        Args:
            features (tensor, float32): A 4D tensor that holds the features
                                        produced by the encoder module.
        """
        print(features.get_shape())
        if self._atflag:
            # features = tfa.layers.MultiHeadAttention(128,12)(features,mask)
            features = tf.concat([secmask,mask,features],
                    axis=self._channel_axis)
        else:
            if self.MMFlag:
                if self._atflag:
                    context = self._iAFFas.forward(features,mask)
                else:
                    context = tf.concat([mask,features],
                    axis=self._channel_axis)
                features = context
            elif self.DMFlag:
                if self.shareFlag:
                    context = tf.concat([mask, features], axis=self._channel_axis)
                else:
                    if self._atflag:
                        context = tf.concat([mask, secmask, features], axis=self._channel_axis)
                    else:
                        context = tf.concat([mask, secmask, features], axis=self._channel_axis)
                features = context

        with tf.variable_scope('Gen'):

            branch1 = tf.layers.conv2d(features, 256, 1,
                                    padding="same",
                                    activation=tf.nn.relu,
                                    data_format=self._data_format,
                                    name="aspp/conv1_1")

            branch2 = tf.layers.conv2d(features, 256, 3,
                                    padding="same",
                                    activation=tf.nn.relu,
                                    dilation_rate=4,
                                    data_format=self._data_format,
                                    name="aspp/conv1_2")

            branch3 = tf.layers.conv2d(features, 256, 3,
                                    padding="same",
                                    activation=tf.nn.relu,
                                    dilation_rate=8,
                                    data_format=self._data_format,
                                    name="aspp/conv1_3")

            branch4 = tf.layers.conv2d(features, 256, 3,
                                    padding="same",
                                    activation=tf.nn.relu,
                                    dilation_rate=12,
                                    data_format=self._data_format,
                                    name="aspp/conv1_4")

            branch5 = tf.reduce_mean(features,
                                    axis=self._dims_axis,
                                    keepdims=True)

            branch5 = tf.layers.conv2d(branch5, 256, 1,
                                    padding="valid",
                                    activation=tf.nn.relu,
                                    data_format=self._data_format,
                                    name="aspp/conv1_5")

            shape = tf.shape(features)

            branch5 = self._upsample(branch5, shape, 1)

            if self.MMFlag:
                # context = tf.concat([mask,branch1, branch2, branch3, branch4, branch5],
                #     axis=self._channel_axis)
                context = tf.concat([branch1, branch2, branch3, branch4, branch5],
                                axis=self._channel_axis)

            elif self.DMFlag:
                if self.shareFlag:
                    # context = tf.concat([mask, branch1, branch2, branch3, branch4, branch5], axis=self._channel_axis)
                    context = tf.concat([branch1, branch2, branch3, branch4, branch5],
                                axis=self._channel_axis)
                else:
                    context = tf.concat([branch1, branch2, branch3, branch4, branch5],
                                axis=self._channel_axis)
                    # context = tf.concat([mask, secmask, branch1, branch2, branch3, branch4, branch5], axis=self._channel_axis)
            else:
                context = tf.concat([branch1, branch2, branch3, branch4, branch5],
                                axis=self._channel_axis)

            aspp_output = tf.layers.conv2d(context, 256, 1,
                                        padding="same",
                                        activation=tf.nn.relu,
                                        data_format=self._data_format,
                                        name="aspp/conv2")
            self._output = aspp_output

    def _decoder(self, features):
        """The decoder model applies a series of 3 upsampling blocks that each
           performs bilinear upsampling followed by a 3x3 convolution to avoid
           checkerboard artifacts in the image space. Unlike all other layers,
           the output of the model is not modified by a ReLU.

        Args:
            features (tensor, float32): A 4D tensor that holds the features
                                        produced by the ASPP module.
        """

        shape = tf.shape(features)

        with tf.variable_scope('Gen'):

            layer1 = self._upsample(features, shape, 2)

            layer2 = tf.layers.conv2d(layer1, 128, 3,
                                    padding="same",
                                    activation=tf.nn.relu,
                                    data_format=self._data_format,
                                    name="decoder/conv1")

            shape = tf.shape(layer2)

            layer3 = self._upsample(layer2, shape, 2)

            layer4 = tf.layers.conv2d(layer3, 64, 3,
                                    padding="same",
                                    activation=tf.nn.relu,
                                    data_format=self._data_format,
                                    name="decoder/conv2")

            shape = tf.shape(layer4)

            layer5 = self._upsample(layer4, shape, 2)

            layer6 = tf.layers.conv2d(layer5, 32, 3,
                                    padding="same",
                                    activation=tf.nn.relu,
                                    data_format=self._data_format,
                                    name="decoder/conv3")

            decoder_output = tf.layers.conv2d(layer6, 1, 3,
                                            padding="same",
                                            data_format=self._data_format,
                                            name="decoder/conv4")

            if self._data_format == "channels_first":
                decoder_output = tf.transpose(decoder_output, (0, 2, 3, 1))

            self._output = decoder_output
        



    def _upsample(self, stack, shape, factor):
        """This function resizes the input to a desired shape via the
           bilinear upsampling method.

        Args:
            stack (tensor, float32): A 4D tensor with the function input.
            shape (tensor, int32): A 1D tensor with the reference shape.
            factor (scalar, int): An integer denoting the upsampling factor.

        Returns:
            tensor, float32: A 4D tensor that holds the activations after
                             bilinear upsampling of the input.
        """

        if self._data_format == "channels_first":
            stack = tf.transpose(stack, (0, 2, 3, 1))

        stack = tf.image.resize_bilinear(stack, (shape[self._dims_axis[0]] * factor,
                                                 shape[self._dims_axis[1]] * factor))

        if self._data_format == "channels_first":
            stack = tf.transpose(stack, (0, 3, 1, 2))

        return stack

    def _normalize(self, maps, eps=1e-7):
        """This function normalizes the output values to a range
           between 0 and 1 per saliency map.

        Args:
            maps (tensor, float32): A 4D tensor that holds the model output.
            eps (scalar, float, optional): A small factor to avoid numerical
                                           instabilities. Defaults to 1e-7.
        """

        min_per_image = tf.reduce_min(maps, axis=(1, 2, 3), keep_dims=True)
        maps -= min_per_image

        max_per_image = tf.reduce_max(maps, axis=(1, 2, 3), keep_dims=True)
        maps = tf.divide(maps, eps + max_per_image, name="output")

        self._output = maps

    def _pretraining(self):
        """The first 26 variables of the model here are based on the VGG16
           network. Therefore, their names are matched to the ones of the
           pretrained VGG16 checkpoint for correct initialization.
        """

        for var in tf.global_variables()[2:26]:
            print(var.name)
            key = var.name.split("/", 1)[1]
            key = key.replace("kernel:0", "weights")
            key = key.replace("bias:0", "biases")
            self._mapping[key] = var

    def forward(self, images):
        """Public method to forward RGB images through the whole network
           architecture and retrieve the resulting output.

        Args:
            images (tensor, float32): A 4D tensor that holds the values of the
                                      raw input images.

        Returns:
            tensor, float32: A 4D tensor that holds the values of the
                             predicted saliency maps.
        """

        if self.MMFlag:
            img = images[:,:,:,0:3]
            mask = images[:,:,:,3]
            mask = tf.expand_dims(mask, axis=-1)
            el06,el10,el18 = self._encoder(img,mask_flag=False)
            ml06,ml10,ml18 = self._maskencoder(mask)
            # # f06 = self._iAFF06.forward(el06, ml06)
            f10 = self._iAFF10.forward(el10, ml10)
            f18 = self._iAFF18.forward(el18,ml18)
            # # f = self._iAFF06.forward(f10,f18)
            f = tf.concat([f10,f18], axis=self._channel_axis)

            self._aspp(self._output, self._maskoutput,f)
            # self._aspp(self._output, self._maskoutput)
            self._decoder(self._output)
            self._normalize(self._output)
        elif self.DMFlag:
            if self.shareFlag:
                if self._atflag:
                    img = images[:,:,:,0:3]
                    mask = images[:,:,:,3:5]
                    # mask = tf.expand_dims(mask, axis=-1)
                    el06,el10,el18 = self._encoder(img,mask_flag=False)
                    ml06,ml10,ml18 = self._maskencoder(mask)
                    # f06 = self._iAFF06.forward(el06, ml06)
                    f10 = self._iAFF10.forward(el10, ml10)
                    f18 = self._iAFF18.forward(el18,ml18)
                    # f = self._iAFF06.forward(f10,f18)
                    f = tf.concat([f10,f18], axis=self._channel_axis)

                    self._aspp(self._output, self._maskoutput, f)
                    self._decoder(self._output)
                    self._normalize(self._output)
                else:
                    img = images[:,:,:,0:3]
                    mask = images[:,:,:,3:5]
                    # mask = tf.expand_dims(mask, axis=-1)
                    self._encoder(img,mask_flag=False)
                    self._maskencoder(mask)
                    self._aspp(self._output, self._maskoutput)
                    self._decoder(self._output)
                    self._normalize(self._output)
            else:
                img = images[:,:,:,0:3]
                mask = images[:,:,:,3]
                mask = tf.expand_dims(mask, axis=-1)
                secmask = images[:,:,:,4]
                secmask = tf.expand_dims(secmask, axis=-1)
                self._encoder(img,mask_flag=False)
                self._maskencoder(mask)
                self._secondmaskencoder(secmask)
                # tmp_f10 = self._iAFF10.forward(el10, iml10)
                # f10 = self._iAFF06.forward(tmp_f10, tml10)
                # tmp_f18 = self._iAFF10.forward(el18, iml18)
                # f18 = self._iAFF06.forward(tmp_f18, tml18)
                self._aspp(self._output, self._maskoutput, self._secondmaskoutput)
                self._decoder(self._output)
                self._normalize(self._output)
            

        else:
            self._encoder(images,mask_flag=False)
            self._aspp(self._output)
            self._decoder(self._output)
            self._normalize(self._output)

        return self._output

    def train(self, ground_truth, predicted_maps, learning_rate, gan_flag = False, disc_error = None, alpha = 0.0):
        """Public method to define the loss function and optimization
           algorithm for training the model.

        Args:
            ground_truth (tensor, float32): A 4D tensor with the ground truth.
            predicted_maps (tensor, float32): A 4D tensor with the predictions.
            learning_rate (scalar, float): Defines the learning rate.

        Returns:
            object: The optimizer element used to train the model.
            tensor, float32: A 0D tensor that holds the averaged error.
        """
        if gan_flag:
            error = tf.keras.losses.BinaryCrossentropy(from_logits=True)(disc_error, tf.ones_like(disc_error)) + alpha * loss.kld(ground_truth, predicted_maps)
            optimizer = tf.train.AdamOptimizer(learning_rate)
        else:
            error = loss.kld(ground_truth, predicted_maps)
            optimizer = tf.train.AdamOptimizer(learning_rate)
            optimizer = optimizer.minimize(error)

        return optimizer, error

    def save(self, saver, sess, dataset, path, device, newMMFlag = False, DMFlag = False, epoch = ""):
        """This saves a model checkpoint to disk and creates
           the folder if it doesn't exist yet.

        Args:
            saver (object): An object for saving the model.
            sess (object): The current TF training session.
            path (str): The path used for saving the model.
            device (str): Represents either "cpu" or "gpu".
        """

        os.makedirs(path, exist_ok=True)
        if newMMFlag:
            saver.save(sess, path + "model_%s_%s_newMM%s.ckpt" % (dataset, device, epoch),
                   write_meta_graph=False, write_state=False)
        elif DMFlag:
            saver.save(sess, path + "model_%s_%s_newDM%s.ckpt" % (dataset, device, epoch),
                   write_meta_graph=False, write_state=False)
        else:
            saver.save(sess, path + "model_%s_%s%s.ckpt" % (dataset, device, epoch),
                   write_meta_graph=False, write_state=False)

    def restore(self, sess, dataset, paths, device, ganflag=False):
        """This function allows continued training from a prior checkpoint and
           training from scratch with the pretrained VGG16 weights. In case the
           dataset is either CAT2000 or MIT1003, a prior checkpoint based on
           the SALICON dataset is required.

        Args:
            sess (object): The current TF training session.
            dataset ([type]): The dataset used for training.
            paths (dict, str): A dictionary with all path elements.
            device (str): Represents either "cpu" or "gpu".

        Returns:
            object: A saver object for saving the model.
        """
        VGG16Flag = True

        model_name = "model_%s_%s" % (dataset, device)
        salicon_name = "model_salicon_%s" % device
        vgg16_name = "vgg16_hybrid"
        # fintuning之后继续训练添加mask的model_eyevedowithdoublemask_gpu_newDM
        eyevedo_name = "model_gazemining_%s" %device

        ext1 = ".ckpt.data-00000-of-00001"
        ext2 = ".ckpt.index"

        saver = tf.train.Saver()

        if self.MMFlag or self.shareFlag:
            VGG16Map = {}
            otherMap = {}
            replaceList = [1,1,2,2,2,2,3,3,3,3,3,3,4,4]
            # 加载maskencoder 参数
            if VGG16Flag:
                for var in tf.global_variables():
                    if var.name.split('/')[0] == 'maskconv' and "conv1_1" not in var.name:
                        key = var.name.split("/", 1)[1]
                        if 'Adam' not in key:
                            key = key.replace("kernel:0", "weights")
                            key = key.replace("bias:0", "biases")
                            VGG16Map[key] = var
                loader = tf.train.Saver(VGG16Map)
                loader.restore(sess, paths["weights"] + vgg16_name + ".ckpt")
            else:
                count = 0
                for var in tf.global_variables():
                    if var.name.split('/')[0] == 'maskconv' and "conv1_1" not in var.name:
                        key = var.name.split("/", 1)[1]
                        if 'Adam' not in key:
                            # key = key.replace("kernel:0", "weights")
                            # key = key.replace("bias:0", "biases")
                            key = var.name.replace('maskconv', "conv" + str(replaceList[count]))
                            key = key.replace(':0','')
                            VGG16Map[key] = var
                            count += 1
                loader = tf.train.Saver(VGG16Map)
                chkp.print_tensors_in_checkpoint_file(paths["latest"] + eyevedo_name + ".ckpt", tensor_name=None, all_tensors = False)
                loader.restore(sess, paths["latest"] + eyevedo_name + ".ckpt")

            # 加载autoencoder fintuning参数
            variables = tf.contrib.framework.get_variables_to_restore()
            vars_to_restore = [v for v in variables]
            del_index = []
            GANmap = {}
            # chkp.print_tensors_in_checkpoint_file(paths["weights"] + vgg16_name + ".ckpt", tensor_name=None, all_tensors = False)
            # for idx,it in enumerate(vars_to_restore):
            #     print(it.name)
            #     if it.name.split('/')[0] == 'maskconv' or "aspp/conv2" in it.name:
            #         del_index.append(idx)
            #         # restore_vars.remove(it)
            # vars_to_restore = [vars_to_restore[i] for i in range(len(vars_to_restore)) if i not in del_index]
            for var in vars_to_restore:
                print(var.name)
                if 'Adam' not in var.name and 'beta' not in var.name and 'maskconv' not in var.name and "aspp" not in var.name and "AFF" not in var.name and "batch_normalization" not in var.name and "maskBN" not in var.name:
                    if "Gen" in var.name:
                        key = var.name.split('/',1)[1]
                        key = "Gen/" + key.replace(':0','')
                        GANmap[key] = var
                    else:
                        GANmap[var.name] = var
            loader = tf.train.Saver(GANmap) 


            # saver = tf.train.Saver(vars_to_restore)
            chkp.print_tensors_in_checkpoint_file(paths["latest"] + eyevedo_name + ".ckpt", tensor_name=None, all_tensors = False)
            loader.restore(sess, paths["latest"] + eyevedo_name + ".ckpt")
            return tf.train.Saver()

        elif self.DMFlag:
            VGG16Map = {}
            otherMap = {}
            replaceList = [1,1,2,2,2,2,3,3,3,3,3,3,4,4]
            # 加载maskencoder 参数
            if VGG16Flag:
                for var in tf.global_variables():
                    if var.name.split('/')[0] == 'maskconv' and "conv1_1" not in var.name:
                        key = var.name.split("/", 1)[1]
                        if 'Adam' not in key:
                            key = key.replace("kernel:0", "weights")
                            key = key.replace("bias:0", "biases")
                            VGG16Map[key] = var
                loader = tf.train.Saver(VGG16Map)
                loader.restore(sess, paths["weights"] + vgg16_name + ".ckpt")
                VGG16Map = {}
                for var in tf.global_variables():
                    if var.name.split('/')[0] == 'secmaskconv' and "conv1_1" not in var.name:
                        key = var.name.split("/", 1)[1]
                        if 'Adam' not in key:
                            key = key.replace("kernel:0", "weights")
                            key = key.replace("bias:0", "biases")
                            VGG16Map[key] = var
                loader = tf.train.Saver(VGG16Map)
                loader.restore(sess, paths["weights"] + vgg16_name + ".ckpt")
            else:
                count = 0
                for var in tf.global_variables():
                    if var.name.split('/')[0] == 'secmaskconv' and "conv1_1" not in var.name:
                        key = var.name.split("/", 1)[1]
                        if 'Adam' not in key:
                            # key = key.replace("kernel:0", "weights")
                            # key = key.replace("bias:0", "biases")
                            key = var.name.replace('secmaskconv', "conv" + str(replaceList[count]))
                            key = key.replace(':0','')
                            VGG16Map[key] = var
                            count += 1
                loader = tf.train.Saver(VGG16Map)
                chkp.print_tensors_in_checkpoint_file(paths["latest"] + eyevedo_name + ".ckpt", tensor_name=None, all_tensors = False)
                loader.restore(sess, paths["latest"] + eyevedo_name + ".ckpt")

            # 加载autoencoder fintuning参数
            variables = tf.contrib.framework.get_variables_to_restore()
            vars_to_restore = [v for v in variables]
            # del_index = []
            # # chkp.print_tensors_in_checkpoint_file(paths["weights"] + vgg16_name + ".ckpt", tensor_name=None, all_tensors = False)
            # for idx,it in enumerate(vars_to_restore):
            #     if it.name.split('/')[0] == 'maskconv' or "aspp/conv2" in it.name or it.name.split('/')[0] == 'secmaskconv':
            #         del_index.append(idx)
            #         # restore_vars.remove(it)
            # vars_to_restore = [vars_to_restore[i] for i in range(len(vars_to_restore)) if i not in del_index]
            GANmap = {}
            for var in vars_to_restore:
                print(var.name)
                if 'Adam' not in var.name and 'beta' not in var.name and 'maskconv' not in var.name and "aspp" not in var.name and "AFF" not in var.name and "batch_normalization" not in var.name and "maskBN" not in var.name:
                    if "Gen" in var.name:
                        key = var.name.split('/',1)[1]
                        key = 'Gen/' + key.replace(':0','')
                        GANmap[key] = var
                    else:
                        GANmap[var.name] = var
            loader = tf.train.Saver(GANmap) 


            # saver = tf.train.Saver(vars_to_restore)
            loader.restore(sess, paths["latest"] + eyevedo_name + ".ckpt")
            return tf.train.Saver()

            
        # 查看所有变量：
        # GANmap = {}
        # variables = tf.contrib.framework.get_variables_to_restore()
        # vars_to_restore = [v for v in variables]
        # for var in vars_to_restore:
        #     print(var.name)
        #     if 'Adam' not in var.name:
        #         if "Gen" in var.name:
        #             key = var.name.split('/',1)[1]
        #             GANmap[key] = var
        #         else:
        #             GANmap[var.name] = var
        
        

        if os.path.isfile(paths["latest"] + model_name + ext1) and \
           os.path.isfile(paths["latest"] + model_name + ext2):
            print(paths["latest"] + model_name + ".ckpt")
            # gan的话第一个预训练要用下面的训练好用上面的，合并一个循环应该
            chkp.print_tensors_in_checkpoint_file(paths["latest"] + model_name + ".ckpt", tensor_name=None, all_tensors = False)
            loadermap = {}
            for var in tf.global_variables():
                if  "beta1_power" not in var.name and "beta2_power" not in var.name and "disc" not in var.name:
                    key = var.name.split("/", 1)[1]
                    if 'Adam' not in key:
                        loadermap[var.name.split(":")[0]] = var
            loader = tf.train.Saver(loadermap)
            loader.restore(sess, paths["latest"] + model_name + ".ckpt")
            # GANmap = {}
            # variables = tf.contrib.framework.get_variables_to_restore()
            # vars_to_restore = [v for v in variables]
            # for var in vars_to_restore:
            #     print(var.name)
            #     if 'Adam' not in var.name and 'beta' not in var.name:
            #         if "Gen" in var.name:
            #             key = var.name.split('/',1)[1]
            #             key = key.replace(':0','')
            #             GANmap[key] = var
            #         else:
            #             GANmap[var.name] = var
            # loader = tf.train.Saver(GANmap)      
            # chkp.print_tensors_in_checkpoint_file(paths["latest"] + model_name + ".ckpt", tensor_name=None, all_tensors = False)
            # loader.restore(sess, paths["latest"] + model_name + ".ckpt")
            # saver.restore(sess, paths["latest"] + model_name + ".ckpt")
        elif dataset in ("mit1003", "cat2000", "dutomron",
                         "pascals", "osie", "fiwi", "gazemining", "eyevedo", "alldataset"):
            if os.path.isfile(paths["best"] + salicon_name + ext1) and \
               os.path.isfile(paths["best"] + salicon_name + ext2):
                GANmap = {}
                variables = tf.contrib.framework.get_variables_to_restore()
                vars_to_restore = [v for v in variables]
                for var in vars_to_restore:
                    print(var.name)
                    if 'Adam' not in var.name and 'beta' not in var.name:
                        if "Gen" in var.name:
                            key = var.name.split('/',1)[1]
                            key = key.replace(':0','')
                            GANmap[key] = var
                        else:
                            GANmap[var.name] = var
                loader = tf.train.Saver(GANmap)      
                chkp.print_tensors_in_checkpoint_file(paths["best"] + salicon_name + ".ckpt", tensor_name=None, all_tensors = False)
                loader.restore(sess, paths["best"] + salicon_name + ".ckpt")
            else:
                raise FileNotFoundError("Train model on SALICON first")
        # 添加判断fintuning之后训练
        elif os.path.isfile(paths["latest"] + eyevedo_name + ext1) and \
           os.path.isfile(paths["latest"] + eyevedo_name + ext2):

            chkp.print_tensors_in_checkpoint_file(paths["latest"] + eyevedo_name + ".ckpt", tensor_name=None, all_tensors = False)
            variables = tf.contrib.framework.get_variables_to_restore()
            # 加载finetuning模型 去除前两个参数
            vars_to_restore = [v for v in variables]
            GANmap = {}
            for var in vars_to_restore:
                print(var.name)
                if 'Adam' not in var.name and 'beta' not in var.name:
                    if "Gen" in var.name:
                        key = var.name.split('/',1)[1]
                        key = key.replace(':0','')
                        GANmap[key] = var
                    else:
                        GANmap[var.name] = var
                loader = tf.train.Saver(GANmap) 
            GANmap.pop('conv1/conv1_1/kernel')
            GANmap.pop('conv1/conv1_1/bias')
            # vars_to_restore.pop(0)
            # vars_to_restore.pop(0)
            loader = tf.train.Saver(GANmap)
            loader.restore(sess, paths["latest"] + eyevedo_name + ".ckpt")
            # self._pretraining()
            # loader = tf.train.Saver(self._mapping)
            # loader.restore(sess, paths["latest"] + eyevedo_name + ".ckpt")
            

        else:
            if not (os.path.isfile(paths["weights"] + vgg16_name + ext1) or
                    os.path.isfile(paths["weights"] + vgg16_name + ext2)):
                download.download_pretrained_weights(paths["weights"],
                                                     "vgg16_hybrid")
            self._pretraining()

            loader = tf.train.Saver(self._mapping)
            loader.restore(sess, paths["weights"] + vgg16_name + ".ckpt")

        return saver

    def optimize(self, sess, dataset, path, device, newMMFlag=False, DMFlag=False, epoch = ""):
        """The best performing model is frozen, optimized for inference
           by removing unneeded training operations, and written to disk.

        Args:
            sess (object): The current TF training session.
            path (str): The path used for saving the model.
            device (str): Represents either "cpu" or "gpu".

        .. seealso:: https://bit.ly/2VBBdqQ and https://bit.ly/2W7YqBa
        """
        if newMMFlag:
            model_name = "model_%s_%s_newMM%s" % (dataset, device, epoch)
        elif DMFlag:
            model_name = "model_%s_%s_newDM%s" % (dataset, device, epoch)
        else:
            model_name = "model_%s_%s%s" % (dataset, device, epoch)
        model_path = path + model_name

        tf.train.write_graph(sess.graph.as_graph_def(),
                             path, model_name + ".pbtxt")

        freeze_graph.freeze_graph(model_path + ".pbtxt", "", False,
                                  model_path + ".ckpt", "output",
                                  "save/restore_all", "save/Const:0",
                                  model_path + ".pb", True, "")

        os.remove(model_path + ".pbtxt")

        graph_def = tf.GraphDef()

        with tf.gfile.Open(model_path + ".pb", "rb") as file:
            graph_def.ParseFromString(file.read())

        transforms = ["remove_nodes(op=Identity)",
                      "merge_duplicate_nodes",
                      "strip_unused_nodes",
                      "fold_constants(ignore_errors=true)"]

        optimized_graph_def = TransformGraph(graph_def,
                                             ["input"],
                                             ["output"],
                                             transforms)

        tf.train.write_graph(optimized_graph_def,
                             logdir=path,
                             as_text=False,
                             name=model_name + ".pb")
