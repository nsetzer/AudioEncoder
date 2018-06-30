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

def parseArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument("input_graph", type=str,
        help="TensorFlow \'GraphDef\' file to load.")
    parser.add_argument("input_checkpoint", type=str,
        help="TensorFlow variables file to load.")
    parser.add_argument("output_graph", type=str, default="",
        help="Output \'GraphDef\' file name.")

    parser.add_argument("--blacklist", type=str, default="",
        help="Variables which should not be converted to constants.")
    parser.add_argument("--output_node_names", type=str, default="",
        help="The name of the output nodes, comma separated.")

    args = parser.parse_args()

    return args

def freeze(input_graph, output_graph, checkpoint, node_names, blacklist=""):

    return freeze_graph(input_graph, "", False, checkpoint,
        node_names, "save/restore_all", "save/Const:0",
        output_graph, True, "",
        variable_names_blacklist=blacklist)

def main():
    args = parseArgs()

    freeze(args.input_graph, args.output_graph,
        args.input_checkpoint, args.output_node_names, args.blacklist)

