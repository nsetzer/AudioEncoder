#!/usr/bin/env python
##======================================================================
## (C) Copyright 2018 by Cogito Corporation
## CONFIDENTIAL.
##----------------------------------------------------------------------
##
## Description:
##   Light wrapper around the freeze_graph tool in order to implement
##   a slight modification that forces TensorFlow to load
##   config.data.dataset ops.
##======================================================================

import argparse
import tensorflow as tf

from tensorflow.python.tools.freeze_graph import freeze_graph

def listVariablesInGraph(filename):
    with open(filename, "w") as f:
        f.write('-------------------------------------------------\n')
        f.write('VARIABLES IN GRAPH\n')
        f.write('-------------------------------------------------\n\n')
        for n in tf.get_default_graph().as_graph_def().node:
            f.write("{0}\n".format(n.name))
        f.write('-------------------------------------------------\n\n')

parser = argparse.ArgumentParser()
parser.register("type", "bool", lambda v: v.lower() == "true")
parser.add_argument(
    "--input_graph",
    type=str,
    default="",
    help="TensorFlow \'GraphDef\' file to load.")
parser.add_argument(
    "--input_checkpoint",
    type=str,
    default="",
    help="TensorFlow variables file to load.")
parser.add_argument(
    "--output_graph",
    type=str,
    default="",
    help="Output \'GraphDef\' file name.")
parser.add_argument(
    "--vars_to_vary",
    type=str,
    default="",
    help="Variables which are allowed to vary within a TF session.")
parser.add_argument(
    "--output_node_names",
    type=str,
    default="",
    help="The name of the output nodes, comma separated.")

FLAGS, unparsed = parser.parse_known_args()

freeze_graph(FLAGS.input_graph,
             "",
             False,
             FLAGS.input_checkpoint,
             FLAGS.output_node_names,
             "save/restore_all",
             "save/Const:0",
             FLAGS.output_graph,
             True,
             "",
             variable_names_blacklist=FLAGS.vars_to_vary)
