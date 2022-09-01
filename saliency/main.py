import argparse
from asyncio import FastChildWatcher
from cmath import isnan
import os
from telnetlib import DM
import cv2
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf

import config
import data
import download
import model
import utils
import loss

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR) 
tf.logging.set_verbosity(tf.logging.ERROR)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1" #(or "1" or "2")

def grad_cam(name, gt, oriSize, filename):

    resList = []
    draw_list = []
    res = np.zeros((216,384))
    for tensor_name in tf.contrib.graph_editor.get_tensors(tf.get_default_graph()):
        print(tensor_name)
        if tensor_name.name.endswith(name) and "AFF" not in tensor_name.name and "aspp" not in tensor_name.name and "conv5" not in tensor_name.name:
            draw_list.append(tensor_name)
    
    logits = tf.get_default_graph().get_tensor_by_name('import/output:0')
    ls = loss.kld(gt,logits)
    for count,  it in enumerate(draw_list):
        itTensor = tf.get_default_graph().get_tensor_by_name(it.name)
        grads = tf.gradients(ls,itTensor)[0]
        castConvOutput = tf.cast(itTensor,tf.float32)
        castGrads = tf.cast(grads,tf.float32)
        guidedGrads = castConvOutput * castGrads * grads
        weights = tf.reduce_mean(guidedGrads,axis=(0,1))
        cam = tf.reduce_sum(tf.multiply(weights, itTensor), axis=1)
        cam_np = cam.eval()
        print(cam_np.shape)
        heatmap = np.squeeze(cam_np)
        print(heatmap.shape)
        heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))
        heatmap *= 255
        heatmap = cv2.resize(heatmap, oriSize)
        heatmap[np.isnan(heatmap)] = 0
        resList.append(heatmap)
    
    for it in resList:
        print(it.shape)
    
    # res/=count

            
    fused = np.mean(np.array(resList), axis=0)
    fused = (fused - np.min(fused)) / (np.max(fused) - np.min(fused))
    fused *= 255
    # fused = cv2.applyColorMap(fused,cv2.COLORMAP_JET)
    cv2.imwrite("/workspace/saliency/results/test_gc/" + filename,fused.astype(np.int32))


def define_paths(current_path, args):
    """A helper function to define all relevant path elements for the
       locations of data, weights, and the results from either training
       or testing a model.

    Args:
        current_path (str): The absolute path string of this script.
        args (object): A namespace object with values from command line.

    Returns:
        dict: A dictionary with all path elements.
    """

    if os.path.isfile(args.path):
        data_path = args.path
    else:
        data_path = os.path.join(args.path, "")
    # 这里修改
    results_path = current_path + "/results/"
    weights_path = current_path + "/weights/"

    history_path = results_path + "history/"
    images_path = results_path + "images/"
    ckpts_path = results_path + "ckpts/"

    best_path = ckpts_path + "best/"
    latest_path = ckpts_path + "latest/"

    if args.phase == "train":
        if args.data not in data_path:
            data_path += args.data + "/"

    paths = {
        "data": data_path,
        "history": history_path,
        "images": images_path,
        "best": best_path,
        "latest": latest_path,
        "weights": weights_path
    }

    return paths

def train_gan_model(dataset, paths, device, alpha=0.7, k=5):

    MMFlag = False
    DMFlag = False
    shareFlag = False
 
    iterator = data.get_dataset_iterator("train", dataset, paths["data"])


    next_element, train_init_op, valid_init_op = iterator

    input_images, ground_truths = next_element[:2]


    input_plhd = tf.placeholder_with_default(input_images,
                                            #  (None, None, None, 5),
                                            #  (None, None, None, 4),
                                             (None, None, None, 3),
                                             name="input")
    msi_net = model.MSINET(MMFlag=MMFlag, DMFlag=DMFlag,shareflag=shareFlag)
    # discriminator = model.DISCRIMINATOR(MMFlag=MMFlag, DMFlag=DMFlag,shareflag=shareFlag)

    fake_map = msi_net.forward(input_plhd)
    # real_out, fake_out = discriminator(ground_truths, fake_map)

    ed_optimizer, ed_loss = msi_net.train(ground_truths, fake_map,
                                    3 * config.PARAMS["learning_rate"])
    # disc_optinizer, disc_loss = discriminator.train(real_out, fake_out, config.PARAMS["learning_rate"])

    n_train_data = getattr(data, dataset.upper()).n_train
    n_valid_data = getattr(data, dataset.upper()).n_valid

    n_train_batches = int(np.ceil(n_train_data / config.PARAMS["batch_size"])-1)
    n_valid_batches = int(np.ceil(n_valid_data / config.PARAMS["batch_size"])-1)

    gen_history = utils.History(n_train_batches,
                            n_valid_batches,
                            dataset,
                            paths["history"],
                            device,
                            prefix="gen_")
    dis_history = utils.History(n_train_batches,
                        n_valid_batches,
                        dataset,
                        paths["history"],
                        device,
                        prefix="disc_")
    history = utils.History(n_train_batches,
                        n_valid_batches,
                        dataset,
                        paths["history"],
                        device)

    progbar = utils.Progbar(n_train_data,
                            n_train_batches,
                            config.PARAMS["batch_size"],
                            config.PARAMS["n_epochs"],
                            history.prior_epochs)
    salprogbar = utils.Progbar(n_train_data,
                        n_train_batches,
                        config.PARAMS["batch_size"],
                        config.PARAMS["n_epochs"],
                        gen_history.prior_epochs)


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        msi_saver = msi_net.restore(sess, dataset, paths, device)
        # disc_saver = discriminator.restore(sess, dataset, paths, device)
        print(">> Start training on %s..." % dataset.upper())

        print("\033[1;35m ONLY ENCODER-DECODER \033[0m")
        for epoch in range(0):
            sess.run(train_init_op)

            for batch in range(n_train_batches):
                _, error = sess.run([ed_optimizer, ed_loss])

                history.update_train_step(error)
                progbar.update_train_step(batch)

            sess.run(valid_init_op)

            for batch in range(n_valid_batches):
                error = sess.run(ed_loss)

                history.update_valid_step(error)
                progbar.update_valid_step()

            msi_net.save(msi_saver, sess, dataset, paths["latest"], device, newMMFlag=MMFlag, DMFlag=DMFlag)

            

            progbar.write_summary(history.get_mean_train_error(False),
                                  history.get_mean_valid_error(False))
            
            history.save_history()

            if history.valid_history[-1] == min(history.valid_history):
                msi_net.save(msi_saver, sess, dataset, paths["best"], device,newMMFlag=MMFlag, DMFlag=DMFlag)
                msi_net.optimize(sess, dataset, paths["best"], device,newMMFlag=MMFlag, DMFlag=DMFlag)

                print("\tBest model!", flush=True)
            
        msi_net.optimize(sess, dataset, paths["latest"], device,newMMFlag=MMFlag, DMFlag=DMFlag)
    # GAN:
    tf.reset_default_graph()
    iterator = data.get_dataset_iterator("train", dataset, paths["data"])

    next_element, train_init_op, valid_init_op = iterator

    input_images, ground_truths = next_element[:2]

    input_plhd = tf.placeholder_with_default(input_images,
                                            #  (None, None, None, 5),
                                            #  (None, None, None, 4),
                                             (16, 216, 384, 5),
                                             name="input")
    
    msi_net = model.MSINET(MMFlag=MMFlag, DMFlag=DMFlag,shareflag=shareFlag)

    fake_map = msi_net.forward(input_plhd)
    discriminator = model.DISCRIMINATOR(MMFlag=MMFlag, DMFlag=DMFlag,shareflag=shareFlag)
    # real_out, fake_out = discriminator.forward(ground_truths, fake_map)
    real_out = discriminator.forward(ground_truths)
    fake_out = discriminator.forward(fake_map)
    disc_optimizer, disc_loss = discriminator.train(real_out, fake_out, config.PARAMS["learning_rate"])
    ed_optimizer, ed_loss = msi_net.train(ground_truths, fake_map,
                                    3 * config.PARAMS["learning_rate"], gan_flag=True, disc_error=fake_out, alpha=alpha)
    
    gen_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Gen')
    disc_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Disc')
    gen_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope = 'Gen')
    disc_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope = 'Disc')

    # with tf.control_dependencies(disc_ops):
    train_discriminator = disc_optimizer.minimize(disc_loss, var_list=disc_var)
    # with tf.control_dependencies(gen_ops):
    train_generetor = ed_optimizer.minimize(ed_loss, var_list=gen_var)


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        msi_saver = msi_net.restore(sess, dataset, paths, device)
        disc_saver = discriminator.restore(sess, dataset, paths, device)
        print(">> Start training on %s..." % dataset.upper())

        print("\033[1;35m SALGAN START \033[0m")
        for epoch in range(config.PARAMS["n_epochs"]):
            sess.run(train_init_op)
            
            for batch in range(n_train_batches):
                        # sess.run(train_init_op)
                # x_val,y_val = sess.run([input_images,ground_truths])
                # if batch % 2 != 0:
                #     _, g_error = sess.run([train_generetor, ed_loss])
                #     gen_history.update_train_step(g_error)
                # else:
                #     _, d_error = sess.run([train_discriminator, disc_loss])
                #     dis_history.update_train_step(d_error)
                res = sess.run({ "disc":[train_discriminator, disc_loss], "gen":[train_generetor, ed_loss]})
                # loss_r = sess.run(r_loss) 
                # loss_f = sess.run(disc_error)
                # _, g_error = sess.run([train_generetor, ed_loss])
                # _, d_error = sess.run([train_discriminator, disc_loss])
                # f_out = sess.run([real_out, fake_out])
                # fr_loss = sess.run([r_loss,f_loss])
                # l1_out = sess.run(tf.keras.losses.BinaryCrossentropy(from_logits=True)(fake_out, tf.ones_like(fake_out)))
                gen_history.update_train_step(res['gen'][1])
                dis_history.update_train_step(res['disc'][1])
                salprogbar.update_train_step(batch)

            sess.run(valid_init_op)

            for batch in range(n_valid_batches):
                # x_val,y_val = sess.run([input_images,ground_truths])
                # print(x_val.shape)
                # print(y_val.shape)
                valres = sess.run({"gen": ed_loss, "disc": disc_loss })

                gen_history.update_valid_step(valres["gen"])
                dis_history.update_valid_step(valres["disc"])
                salprogbar.update_valid_step()

            msi_net.save(msi_saver, sess, dataset, paths["latest"], device, newMMFlag=MMFlag, DMFlag=DMFlag)
            discriminator.save(disc_saver, sess, dataset, paths["latest"], device, newMMFlag=MMFlag, DMFlag=DMFlag)

            salprogbar.write_summary(gen_history.get_mean_train_error(False),
                        gen_history.get_mean_valid_error(False))
            gen_history.save_history()
            dis_history.save_history()



            if gen_history.valid_history[-1] == min(gen_history.valid_history):
                msi_net.save(msi_saver, sess, dataset, paths["best"], device,newMMFlag=MMFlag, DMFlag=DMFlag)
                msi_net.optimize(sess, dataset, paths["best"], device,newMMFlag=MMFlag, DMFlag=DMFlag)
                # discriminator.save(disc_saver, sess, dataset, paths["best"], device,newMMFlag=MMFlag, DMFlag=DMFlag)
                # discriminator.optimize(sess, dataset, paths["best"], device,newMMFlag=MMFlag, DMFlag=DMFlag)
                print("\tBest model!", flush=True)

            # if epoch % 10 == 0:
            #     msi_net.save(msi_saver, sess, dataset, paths["best"], device,newMMFlag=MMFlag, DMFlag=DMFlag, epoch="_" + str(epoch))
            #     msi_net.optimize(sess, dataset, paths["best"], device,newMMFlag=MMFlag, DMFlag=DMFlag, epoch="_" + str(epoch))
            
        msi_net.optimize(sess, dataset, paths["latest"], device,newMMFlag=MMFlag, DMFlag=DMFlag)
        discriminator.optimize(sess, dataset, paths["latest"], device,newMMFlag=MMFlag, DMFlag=DMFlag)


def train_model(dataset, paths, device):
    """The main function for executing network training. It loads the specified
       dataset iterator, saliency model, and helper classes. Training is then
       performed in a new session by iterating over all batches for a number of
       epochs. After validation on an independent set, the model is saved and
       the training history is updated.

    Args:
        dataset (str): Denotes the dataset to be used during training.
        paths (dict, str): A dictionary with all path elements.
        device (str): Represents either "cpu" or "gpu".
    """
    MMFlag = True
    DMFlag = False
    shareFlag = False
    atFlag = True
 
    iterator = data.get_dataset_iterator("train", dataset, paths["data"])

    next_element, train_init_op, valid_init_op = iterator

    input_images, ground_truths = next_element[:2]

    input_plhd = tf.placeholder_with_default(input_images,
                                            #  (None, None, None, 5),
                                             (8, 216, 384, 4),
                                            #  (None, None, None, 3),
                                             name="input")
    msi_net = model.MSINET(MMFlag=MMFlag, DMFlag=DMFlag,shareflag=shareFlag, attentionFlag=atFlag)

    predicted_maps = msi_net.forward(input_plhd)

    optimizer, loss = msi_net.train(ground_truths, predicted_maps,
                                    config.PARAMS["learning_rate"])

    n_train_data = getattr(data, dataset.upper()).n_train
    n_valid_data = getattr(data, dataset.upper()).n_valid

    n_train_batches = int(np.ceil(n_train_data / config.PARAMS["batch_size"])) -1
    n_valid_batches = int(np.ceil(n_valid_data / config.PARAMS["batch_size"])) -1

    history = utils.History(n_train_batches,
                            n_valid_batches,
                            dataset,
                            paths["history"],
                            device)

    progbar = utils.Progbar(n_train_data,
                            n_train_batches,
                            config.PARAMS["batch_size"],
                            config.PARAMS["n_epochs"],
                            history.prior_epochs)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = msi_net.restore(sess, dataset, paths, device)

        print(">> Start training on %s..." % dataset.upper())

        for epoch in range(config.PARAMS["n_epochs"]):
            sess.run(train_init_op)

            for batch in range(n_train_batches):
                _, error = sess.run([optimizer, loss])

                history.update_train_step(error)
                progbar.update_train_step(batch)

            sess.run(valid_init_op)

            for batch in range(n_valid_batches):
                error = sess.run(loss)

                history.update_valid_step(error)
                progbar.update_valid_step()

            msi_net.save(saver, sess, dataset, paths["latest"], device, newMMFlag=MMFlag, DMFlag=DMFlag)

            history.save_history()

            progbar.write_summary(history.get_mean_train_error(),
                                  history.get_mean_valid_error())

            if history.valid_history[-1] == min(history.valid_history):
                msi_net.save(saver, sess, dataset, paths["best"], device,newMMFlag=MMFlag, DMFlag=DMFlag)
                msi_net.optimize(sess, dataset, paths["best"], device,newMMFlag=MMFlag, DMFlag=DMFlag)

                print("\tBest model!", flush=True)
            
        msi_net.optimize(sess, dataset, paths["latest"], device,newMMFlag=MMFlag, DMFlag=DMFlag)


def test_model(dataset, paths, device):
    """The main function for executing network testing. It loads the specified
       dataset iterator and optimized saliency model. By default, when no model
       checkpoint is found locally, the pretrained weights will be downloaded.
       Testing only works for models trained on the same device as specified in
       the config file.

    Args:
        dataset (str): Denotes the dataset that was used during training.
        paths (dict, str): A dictionary with all path elements.
        device (str): Represents either "cpu" or "gpu".
    """
    label_path = paths["data"] + "/label/"
    iterator = data.get_dataset_iterator("test", dataset, paths["data"])

    next_element, init_op = iterator

    input_images, original_shape, file_path = next_element
    # input_images, ground_truths = next_element[:2]
    print(input_images.shape)

    graph_def = tf.GraphDef()

    model_name = "model_gazeminingwithmask_gpu_newMM.pb" #% (dataset, device)
    # model_name = "model_%s_%s.pb" % (dataset, device)
    print(paths["best"] + model_name)
    if os.path.isfile(paths["best"] + model_name):
        print("HI")
        with tf.gfile.Open(paths["best"] + model_name, "rb") as file:
            graph_def.ParseFromString(file.read())
    else:
        if not os.path.isfile(paths["weights"] + model_name):
            download.download_pretrained_weights(paths["weights"],
                                                 model_name[:-3])

        with tf.gfile.Open(paths["weights"] + model_name, "rb") as file:
            graph_def.ParseFromString(file.read())

    [predicted_maps] = tf.import_graph_def(graph_def,
                                           input_map={"input": input_images},
                                           return_elements=["output:0"])

    jpeg = data.postprocess_saliency_map(predicted_maps[0],
                                         original_shape[0])





    print(">> Start testing with %s %s model..." % (dataset.upper(), device))

    with tf.Session() as sess:
        sess.run(init_op)

        while True:
            try:
                # res = sess.run(original_shape)
                output_file, path = sess.run([jpeg, file_path])
            except tf.errors.OutOfRangeError:
                break

            path = path[0][0].decode("utf-8")
            print(path)
            filename = os.path.basename(path)
            filename = os.path.splitext(filename)[0]
            filename += ".jpeg"

            os.makedirs(paths["images"], exist_ok=True)

            with open(paths["images"] + filename, "wb") as file:
                file.write(output_file)

            # #     # Grad-Cam
            # it_label = label_path + filename.replace('.jpeg','_pureheat.png')
            # img_label = cv2.imread(it_label,cv2.IMREAD_GRAYSCALE)
            # gt = tf.convert_to_tensor(img_label)
            # gt = tf.expand_dims(gt, axis = -1)
            # gt = tf.expand_dims(gt, axis = 0)
            # gt = tf.cast(gt, dtype=tf.float32)
            # gt = tf.image.resize_images(gt,(216,384))
            # grad_cam("Conv2D:0",gt,(384,216),filename)

            # try:
            #     for tensor_name in tf.contrib.graph_editor.get_tensors(tf.get_default_graph()):
            #         print(tensor_name)
            #     # tensor_dict = tf.get_default_graph().get_tensor_by_name("import/aspp/conv2/Conv2D:0")
            #     # tensor_dict = tf.get_default_graph().get_tensor_by_name("import/Gen_1/aspp/conv1_1/Conv2D:0")
            #     affout = tf.get_default_graph().get_tensor_by_name("import/add_7:0")
            #     enout = tf.get_default_graph().get_tensor_by_name("import/Gen/max_pooling2d_4/MaxPool:0")
            #     meout = tf.get_default_graph().get_tensor_by_name("import/maskconv/conv4_1/Conv2D:0")
            #     asppout = tf.get_default_graph().get_tensor_by_name("import/Gen_1/aspp/conv2/Relu:0")
            #     logits = tf.get_default_graph().get_tensor_by_name('import/output:0')
                
            
            #     # tmp = tensor_dict.eval()
            #     # pltmap = np.zeros((27,48))
            #     # for i in range(256):
            #     #     pltmap += tmp[0,i,:,:]

            #     # pltmap = (pltmap - np.min(pltmap)) / (np.max(pltmap) - np.min(pltmap))
            #     # pltmap = pltmap * 255
            #     # cv2.imwrite("/workspace/saliency/results/asppout1/" + filename, pltmap)

            #     # plt.plot(pltmap)
            #     # plt.savefig("/workspace/saliency/test.png")
                
            #     print(path)
            #     it_label = label_path + filename.replace('.jpeg','_pureheat.png')
            #     img_label = cv2.imread(it_label,cv2.IMREAD_GRAYSCALE)
            #     print((logits.shape[1],logits.shape[2]))
            #     w = logits.eval().shape[2]
            #     h = logits.eval().shape[1]
            #     img_label = cv2.resize(img_label,(w,h))
            #     pred = tf.image.decode_jpeg(jpeg,channels = 1)
            #     gt = tf.convert_to_tensor(img_label)
            #     gt = tf.expand_dims(gt, axis = -1)
            #     gt = tf.expand_dims(gt, axis = 0)
            #     pred = tf.expand_dims(pred, axis = -1)
            #     gt = tf.cast(gt, dtype=tf.float32)
            #     pred = tf.cast(gt, dtype = tf.float32)
            #     pred = pred/255.0
            #     # tf.image.resize_images(gt,logits.shape)
            #     ls = loss.kld(gt,logits)
            #     # print(ls_a.eval())
            #     # for tensor_name in tf.contrib.graph_editor.get_tensors(tf.get_default_graph()):
            #     #     print(tensor_name)
            #     # tensor_dict = tf.get_default_graph().get_tensor_by_name("import/aspp/conv2/Conv2D:0")
            #     # print(tensor_dict.shape.eval())
            #     grads = tf.gradients(ls,affout)[0]
            #     grads_en = tf.gradients(ls,enout)[0]
            #     grads_me = tf.gradients(ls,meout)[0]
            #     grads_aspp = tf.gradients(ls,asppout)[0]
            #     castConvOutput = tf.cast(affout,tf.float32)
            #     castConvOutput_en = tf.cast(enout,tf.float32)
            #     castConvOutput_me = tf.cast(meout,tf.float32)
            #     castConvOutput_aspp = tf.cast(asppout,tf.float32)
            #     castGrads = tf.cast(grads,tf.float32)
            #     castGrads_en = tf.cast(grads_en,tf.float32)
            #     castGrads_me = tf.cast(grads_me,tf.float32)
            #     castGrads_aspp = tf.cast(grads_aspp,tf.float32)
            #     guidedGrads = castConvOutput * castGrads * grads
            #     guidedGrads_en = castConvOutput_en * castGrads_en * grads_en
            #     guidedGrads_me = castConvOutput_me * castGrads_me * grads_me
            #     guidedGrads_aspp = castConvOutput_aspp * castGrads_aspp * grads_aspp
            #     weights = tf.reduce_mean(guidedGrads,axis=(0,1))
            #     weights_en = tf.reduce_mean(guidedGrads_en,axis=(0,1))
            #     weights_me = tf.reduce_mean(guidedGrads_me,axis=(0,1))
            #     weights_aspp = tf.reduce_mean(guidedGrads_aspp,axis=(0,1))
            #     # print(weights.eval().shape)
            #     # print(images.eval().shape)
            #     cam = tf.reduce_sum(tf.multiply(weights, affout), axis=1)
            #     cam_en = tf.reduce_sum(tf.multiply(weights_en, enout), axis=1)
            #     cam_me = tf.reduce_sum(tf.multiply(weights_me, meout), axis=1)
            #     cam_aspp = tf.reduce_sum(tf.multiply(weights_aspp, asppout), axis=1)

            #     heatmap = np.squeeze(cam.eval())
            #     heatmap_en = np.squeeze(cam_en.eval())
            #     heatmap_me = np.squeeze(cam_me.eval())
            #     heatmap_aspp = np.squeeze(cam_aspp.eval())

            #     heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))
            #     heatmap *= 255
            #     heatmap_en = (heatmap_en - np.min(heatmap_en)) / (np.max(heatmap_en) - np.min(heatmap_en))
            #     heatmap_en *= 255
            #     heatmap_me = (heatmap_me - np.min(heatmap_me)) / (np.max(heatmap_me) - np.min(heatmap_me))
            #     heatmap_me *= 255
            #     heatmap_aspp = (heatmap_aspp - np.min(heatmap_aspp)) / (np.max(heatmap_aspp) - np.min(heatmap_aspp))
            #     heatmap_aspp *= 255
            #     print(path)
            #     # cv2.imshow("affout",heatmap.astype(int))
            #     # cv2.imshow("affout",heatmap_en.astype(int))
            #     # cv2.imshow("affout",heatmap_me.astype(int))
            #     cv2.imwrite("/workspace/saliency/results/middelout/aff/" + filename, heatmap.astype(int))
            #     cv2.imwrite("/workspace/saliency/results/middelout/en/" + filename, heatmap_en.astype(int))
            #     cv2.imwrite("/workspace/saliency/results/middelout/me/" + filename, heatmap_me.astype(int))
            #     cv2.imwrite("/workspace/saliency/results/middelout/aspp/" + filename, heatmap_aspp.astype(int))
            #     # utils.visualize(jpeg, tensor_dict, gb_grad, ls_p)
            
            # except Exception as e:
            #     print(e)
 




def main():
    """The main function reads the command line arguments, invokes the
       creation of appropriate path variables, and starts the training
       or testing procedure for a model.
    """

    current_path = os.path.dirname(os.path.realpath(__file__))
    default_data_path = current_path + "/data"

    phases_list = ["train", "test"]

    datasets_list = ["salicon", "mit1003", "cat2000",
                     "dutomron", "pascals", "osie", "fiwi", "gazemining", "gazeminingwithmask", "gazeminingwithdoublemask", "eyevedo", "eyevedowithmask", "eyevedowithdoublemask", "alldataset", "alldatasetwithmask", "alldatasetwithdoublemask"]

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("phase", metavar="PHASE", choices=phases_list,
                        help="sets the network phase (allowed: train or test)")

    parser.add_argument("-d", "--data", metavar="DATA",
                        choices=datasets_list, default=datasets_list[0],
                        help="define which dataset will be used for training \
                              or which trained model is used for testing")

    parser.add_argument("-p", "--path", default=default_data_path,
                        help="specify the path where training data will be \
                              downloaded to or test data is stored")

# TODO: 提高了数据，尝试doublemask data改了 ph没改，
    # args = parser.parse_args(['train', '-d', 'gazeminingwithmask'])
    args = parser.parse_args(['test', '-d', 'salicon','-p', '/workspace/saliency/data/gazeminingwithdoublemask/Test/'])
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    print(physical_devices)
    # tf.config.experimental.set_memory_growth(0.75)
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    paths = define_paths(current_path, args)
    print("\n PATH: ", paths)

    if args.phase == "train":
        # train_gan_model(args.data, paths, config.PARAMS["device"])
        train_model(args.data, paths, config.PARAMS["device"])
    elif args.phase == "test":
        test_model(args.data, paths, config.PARAMS["device"])

if __name__ == "__main__":
    # print(tf.__version__)
    main()
