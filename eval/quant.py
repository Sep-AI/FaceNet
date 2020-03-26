import tensorflow as tf
import inception_resnet_v1 as network
from utils import *
import config
import tensorflow.lite as lite


def fix_error():
    gp = tf.get_default_graph().as_graph_def()
    for node in gp.node:
        if node.op == 'RefSwitch':
            node.op = 'Switch'

            for index in range(len(node.input)):
                if 'moving_' in node.input[index]:
                    node.input[index] = node.input[index] + '/read'
        elif node.op == 'AssignSub':

            node.op = 'Sub'
            if 'use_locking' in node.attr: del node.attr['use_locking']
        elif node.op == 'AssignAdd':

            node.op = 'Add'
            if 'use_locking' in node.attr: del node.attr['use_locking']
        elif node.op == 'Assign':
            node.op = 'Identity'

            if 'use_locking' in node.attr: del node.attr['use_locking']
            if 'validate_shape' in node.attr: del node.attr['validate_shape']
            if len(node.input) == 2:
                # input0: ref: Should be from a Variable node. May be uninitialized.
                # input1: value: The value to be assigned to the variable.
                node.input[0] = node.input[1]
                del node.input[1]

def main():
    eval_graph = tf.Graph()
    with eval_graph.as_default():
        inp = tf.placeholder(dtype=tf.float32, shape=[None, config.image_size, config.image_size, 3], name='input')
        # phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')
        phase_train_placeholder = tf.placeholder_with_default(input=False, shape=[], name='phase_train')
        # batch_size_placeholder = tf.placeholder(tf.int32, name='batch_size')
        keep_probability_placeholder = tf.placeholder_with_default(input=1., shape=[], name='keep_probability')
        # keep_probability_placeholder = tf.placeholder(tf.float32, name='keep_probability')
        with tf.variable_scope('quantize'):
            prelogits, _ = network.inference(inp,
                                             keep_probability=keep_probability_placeholder,
                                             phase_train=phase_train_placeholder,
                                             bottleneck_layer_size=config.embedding_size,
                                             weight_decay=config.weight_decay)
            # prelogits, _ = network.inference(inp,
            #                                  1.0,
            #                                  False,
            #                                  bottleneck_layer_size=config.embedding_size,
            #                                  weight_decay=config.weight_decay)
            prelogits = tf.maximum(prelogits, -1e27)

        g = tf.get_default_graph()
        tf.contrib.quantize.create_eval_graph(input_graph=g)
        saver = tf.train.Saver()

        eval_graph.finalize()
        with open('eval.pb', 'w') as f:
            f.write(str(g.as_graph_def()))

    with tf.Session(graph=eval_graph) as session:
        checkpoint = tf.train.latest_checkpoint('/homes/smeshkinfamfard/PycharmProjects/tensorflow-facenet/model/')
        # import pdb;pdb.set_trace()
        saver.restore(session, checkpoint)
        fix_error()
        # fix the input, output, choose types of the weights and activations for the tflite model
        converter = lite.TFLiteConverter.from_session(session, [inp], [prelogits])
        converter.inference_type = tf.uint8
        converter.inference_input_type = tf.uint8
        input_arrays = converter.get_input_arrays()
        converter.quantized_input_stats = {input_arrays[0]: (0., 1.)}

        flatbuffer = converter.convert()

        with open('test.tflite', 'wb') as outfile:
            outfile.write(flatbuffer)
    print('Model successfully converted and saved in the project directory')


if __name__ == '__main__':
    main()
