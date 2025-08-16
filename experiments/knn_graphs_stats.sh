#!/bin/bash

. .venv/bin/activate

EPOCHS=1
HIDDEN_DIM=32
LAYER_NUMBER=1
SELF_LOOP=0
STGI_MODE='s'
MLP_SIZE=32
DATASET="airq_small"
NUM_NODES=36
FRACTION=0.05
LAYER_TYPE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --layer_type)
            LAYER_TYPE="$2"
            shift 2
            ;;
        --self-loop)
            SELF_LOOP=1
            shift
            ;;
        --hidden_dim)
            HIDDEN_DIM="$2"
            shift 2
            ;;
        --layers)
            LAYER_NUMBER="$2"
            shift 2
            ;;
        --mode)
            STGI_MODE="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [[ "$DATASET" == "airq" ]]; then
    NUM_NODES=437
fi

DATE=$(date +%y%m%d)
EXP_DIR="./experiments/results/graphs_stats/knn/"
LOGFILE="${EXP_DIR}${DATE}-knn-experiments.txt"

echo "Running experiments on $DATE" >> "$LOGFILE"

for K in $(seq 0.0 $FRACTION 1.0); do
    echo "Running: -g knn $K -e $EPOCHS -l $LAYER_TYPE" | tee -a "$LOGFILE"
    python -u main.py -d $DATASET -sp $EXP_DIR -sg knn $K -e $EPOCHS -l $LAYER_TYPE \
           -hd $HIDDEN_DIM -ln $LAYER_NUMBER -lr $LR -m $STGI_MODE -sl $SELF_LOOP -gs -dt -v 0 | tee -a "$LOGFILE"
done
