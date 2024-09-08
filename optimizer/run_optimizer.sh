#!/bin/bash

MODEL=alexnet
NODES=2
PROC_PER_NODE=4
NETWORK_FILEPATH=network.json
MEM_CAPACITY=80000000000
TYPE_COMPRESS=1.0
BATCH_RATIO=1.0

python3 optimizer.py \
        -f ../profiler/image_classification/profiles/$MODEL/graph.txt -n $PROC_PER_NODE $NODES \
        -b $NETWORK_FILEPATH -o partitioned/$MODEL --use_memory_constraint -s $MEM_CAPACITY \
        --activation_compression_ratio $TYPE_COMPRESS --minibatch_ratio $BATCH_RATIO --use_recompute
