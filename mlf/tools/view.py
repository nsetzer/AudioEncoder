import tensorflow as tf
from tensorflow.python.platform import gfile
import argparse

def main():

    parser = argparse.ArgumentParser(description="""
        view a frozen model using tensorboard

        tensorboard --logdir=./tmp
    """)
    parser.add_argument('model', type=str,
                        help='the frozen model to process')
    parser.add_argument('logdir', type=str,
                        help='the log directory')

    args = parser.parse_args()

    with tf.Session() as sess:
        with gfile.FastGFile(args.model, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            g_in = tf.import_graph_def(graph_def)
        train_writer = tf.summary.FileWriter(args.logdir)
        train_writer.add_graph(sess.graph)
        train_writer.close()

    print("Success!")
    print("run the following to view this model in tensorboard")
    print("  tensorboard --logdir=\"%s\"" % args.logdir)

if __name__ == '__main__':
    main()