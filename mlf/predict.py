#! cd .. && python36 -m mlf.predict

import json
import os
import time
from collections import defaultdict

import tensorflow as tf
import numpy as np

import SigProc

def load_frozen_graph(frozen_graph_filename, prefix="graph"):
    # load protobuf file
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # import the graphdef as a graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name=prefix)

    return graph

class FrozenModel(object):
    """docstring for FrozenModel"""
    def __init__(self, modelDirectory, modelName):
        super(FrozenModel, self).__init__()

        self._load_graph(modelDirectory, modelName)

    def _load_graph(self, modelDirectory, modelName):
        """
        load a model, which is a pair of a protobuf file and a json file
        which describes the contents of the model

        return a graph, along with a dictionary containing the tensors and
        operations used by the graph
        """

        modelInfoPath = os.path.join(modelDirectory, modelName + ".json")
        modelPath = os.path.join(modelDirectory, modelName + ".pb")

        with open(modelInfoPath, "r") as wf:
            self.info = json.load(wf)

        prefix = "graph"

        graph = load_frozen_graph(modelPath, prefix)

        tensors = {}
        for label, op_name in self.info['tensors'].items():
            tensor = prefix + "/" + op_name
            tensors[label] = graph.get_tensor_by_name(tensor)

        operations = {}
        for label, op_name in self.info['operations'].items():
            operation = prefix + "/" + op_name
            operations[label] = graph.get_operation_by_name(operation)

        #for op in graph.get_operations():
        #    print(op.name)

        self.graph = graph
        self.tensors = tensors
        self.operations = operations

        ######################################################################
        # get parameters used for feature extraction
        procs = []
        for obj in self.info['dataset']['procs']:
            # there is only a single process in this dictionary
            for proc_name, proc_settings in obj.items():
                proc = (getattr(SigProc, proc_name), proc_settings)
                procs.append(proc)

        self.fe_sliceSize = self.info['dataset']['sliceSize']
        self.fe_sliceStep = self.info['dataset']['sliceStep']
        self.classes = self.info['dataset']['classes']
        self.fe_procs = procs

    def extractFrames(self, inFilePath):
        """
        extract frames from an audio file using the exact settings that
        were used to generate the training data
        """
        proc_runner = SigProc.PipelineProcessRunner(self.fe_procs, inFilePath)
        results = proc_runner.run()
        data = results[-1]  # get the process output
        end = data.shape[0] - self.fe_sliceSize + 1
        for i in range(0, end, self.fe_sliceStep):
            yield data[i:i + self.fe_sliceSize]

    def init(self, sess):
        sess.run(self.operations['init'])

    def classifyFrame(self, sess, frame):

        feed_dict = {
            self.tensors['x']: np.reshape(frame, self.tensors['x'].shape)
        }
        # we assume that classes maps to a tensor which produces an
        # argmax. the output should then be an array of integers

        prediction = sess.run(self.tensors['classes'], feed_dict=feed_dict)

        # map argmax integers to
        prediction = [self.classes[i] for i in prediction]

        return prediction

def main():

    inFilePath = "D:\\git\\genres\\rock\\rock.00000.au"
    modelDirectory = "./build/experiment/eval"
    modelName = "frozen_model"

    start = time.time()
    model = FrozenModel(modelDirectory, modelName)

    with tf.Session(graph=model.graph) as sess:
        model.init(sess)

        counts = defaultdict(int)
        for frame in model.extractFrames(inFilePath):
            pred_cls = model.classifyFrame(sess, frame)
            for cls in pred_cls:
                counts[cls] += 1
        print(counts)
    end = time.time()
    print(end - start)

if __name__ == '__main__':
    main()