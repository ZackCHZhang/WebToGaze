import numpy as np
import tensorflow as tf
import tensorflow.contrib.graph_editor as ge

# Load protobuf as graph, given filepath
def load_pb(path_to_pb):
    with tf.gfile.GFile(path_to_pb, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')
        return graph





def second_step(tf_graph,to_convert):
    const_var_name_pairs = []
    with tf_graph.as_default() as g:
        
        for name in to_convert:
            if name != "sub" and name != "input":
                print(name)
                tensor = g.get_tensor_by_name('{}:0'.format(name))
                with tf.Session() as sess:
                    tensor_as_numpy_array = sess.run(tensor)
                var_shape = tensor.get_shape()
                # Give each variable a name that doesn't already exist in the graph
                var_name = '{}_turned_var'.format(name)
                # Create TensorFlow variable initialized by values of original const.
                var = tf.get_variable(name=var_name, dtype='float32', shape=var_shape, initializer=tf.constant_initializer(tensor_as_numpy_array))
                # We want to keep track of our variables names for later.
                const_var_name_pairs.append((name, var_name))

        # At this point, we added a bunch of tf.Variables to the graph, but they're
        # not connected to anything.

        # The magic: we use TF Graph Editor to swap the Constant nodes' outputs with
        # the outputs of our newly created Variables.

        for const_name, var_name in const_var_name_pairs:
            const_op = g.get_operation_by_name(const_name)
            var_reader_op = g.get_operation_by_name(var_name + '/read')
            ge.swap_outputs(ge.sgv(const_op), ge.sgv(var_reader_op))

def thired_step():
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        save_path = tf.train.Saver().save(sess, 'model.ckpt')
        print("Model saved in path: %s" % save_path)

if __name__ == '__main__':
    tf_graph = load_pb('E:/My_MA/saliency/weights/model_salicon_cpu.pb')
    print([n.name for n in tf_graph.as_graph_def().node])
    to_convert = [n.name for n in tf_graph.as_graph_def().node]
    second_step(tf_graph,to_convert)
    thired_step()