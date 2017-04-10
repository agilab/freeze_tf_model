import os
import tensorflow as tf
from tensorflow.python.framework import graph_util

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("output_node_names", "",
                       "Comma-separated output node names.")
tf.flags.DEFINE_string("model_folder", "",
                       "Folder to read saved model from.")
tf.flags.DEFINE_string("output_file", "frozen_model.pb",
                       "File to write serialized pb, in --model_folder.")

def freeze_graph():
    # We retrieve our checkpoint fullpath
    checkpoint = tf.train.get_checkpoint_state(FLAGS.model_folder)
    input_checkpoint = checkpoint.model_checkpoint_path
    
    # We precise the file fullname of our freezed graph
    absolute_model_folder = "/".join(input_checkpoint.split('/')[:-1])
    output_graph = absolute_model_folder + "/" + FLAGS.output_file 

    # We clear devices to allow TensorFlow to control on which device it will load operations
    clear_devices = True
    
    # We import the meta graph and retrieve a Saver
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)

    # We retrieve the protobuf graph definition
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()

    # We start a session and restore the graph weights
    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)

        # We use a built-in TF helper to export variables to constants
        output_graph_def = graph_util.convert_variables_to_constants(
            sess, # The session is used to retrieve the weights
            input_graph_def, # The graph_def is used to retrieve the nodes
            FLAGS.output_node_names.split(",") # The output node names are used to select the usefull nodes
        )

        # Finally we serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))


if __name__ == '__main__':
    assert FLAGS.model_folder, "--model_folder is required"
    assert FLAGS.output_file, "--model_file is required"
    assert FLAGS.output_node_names, "--output_node_name is required"
    freeze_graph()
