
import os, sys
import tensorflow as tf
from google.protobuf import text_format
from tensorflow.python.platform import gfile

def graphdef_to_pbtxt(filename):
    with gfile.FastGFile(filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')
        base, name = os.path.split(filename)
        base = base or "."
        tf.train.write_graph(graph_def, base, name + 'txt', as_text=True)

if __name__ == '__main__':
    graphdef_to_pbtxt(sys.argv[1])