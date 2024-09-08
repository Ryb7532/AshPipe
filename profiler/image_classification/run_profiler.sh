#!/bin/bash

# DATA_DIR=
MODEL=alexnet
BATCH_SIZE=256

if [ "${DATA_DIR}" == "" ]; then
    python3 main.py -s -a $MODEL -b $BATCH_SIZE
else
    python3 main.py --data_dir $DATA_DIR -a $MODEL -b $BATCH_SIZE
fi
