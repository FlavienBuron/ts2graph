#!/bin/bash

. .venv/bin/activate

EPOCHS=1
HIDDEN_DIM=32
LAYER_NUMBER=1
SELF_LOOP=0
USE_TEMPORAL=0
MLP_SIZE=32
DATASET="airq_small"
NUM_NODES=36
FRACTION=0.05

while [[ $# -gt 0 ]]; do
    case "$1" in
        --fraction)
            FRACTION="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
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
        --temporal)
            USE_TEMPORAL=1
            shift
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

if [[ -z "$LR" || "$LR" == "0" ]]; then
    if [[ "$LAYER_NUMBER" -eq 1 ]]; then
        LR=0.005
    else
        LR=0.0005
    fi
fi


DATE=$(date +%y%m%d)
EXP_DIR="./experiments/results/graphs_stats/"
mkdir -p "$EXP_DIR"

# Sweep knn values from 1 to KNN_MAX
for RAD in $(seq 0.0 $FRACTION 1.0); do
    printf -v RAD_FMT "%.2f" "$RAD"
    echo "Running: -g rad $RAD_FMT -e $EPOCHS"
    TIMESTAMP=$(date +%y%m%d_%H%M%S)
    FILENAME="${EXP_DIR}${TIMESTAMP}_${DATASET}_rad_${RAD_FMT}_sl${SELF_LOOP}.json"
    python -u main.py -d $DATASET -sp $FILENAME -sg rad "$RAD_FMT" -e "$EPOCHS" -bs 128 -hd 32 -ln 1 -lr $LR -m s -sl $SELF_LOOP -gs -dt -v 0 | tee -a "$LOGFILE"
done
