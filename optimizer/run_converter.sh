#!/bin/bash

MODEL=alexnet
NODES=2
PROC_PER_NODE=4
GPUS=`expr $NODES \* $PROC_PER_NODE`

echo "Specify stage_to_num_ranks_map (see stage_to_num_ranks_map in output of optimizer.py):"
read $STAGE_RANKS_MAP

if [ ! -e ../runtime/models/$MODEL ]; then
    mkdir ../runtime/models/$MODEL
fi

python3 convert_graph_to_MODEL.py \
        -f partitioned/$MODEL/gpus=$GPUS.txt -n ${MODEL}Partitioned -a $MODEL \
        -o ../runtime/models/$MODEL --stage_to_num_ranks_map $STAGE_RANKS_MAP
