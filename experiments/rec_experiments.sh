#!/bin/bash

. .venv/bin/activate

EPOCHS=30
HIDDEN_DIM=32
LAYER_NUMBER=1
SELF_LOOP=0
STGI_MODE='t'
MLP_SIZE=32
DATASET="airq_small"
NUM_NODES=36
FRACTION=0.05
BATCH_SIZE=128
DECAY="exp"

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
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --decay)
            DECAY="$2"
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
        LR=0.007
    elif [[ "$LAYER_NUMBER" -eq 2 ]]; then
        LR=0.0004
    else
        LR=0.0002
    fi
fi


DATE=$(date +%y%m%d)
EXP_DIR="./experiments/results/rec/ln${LAYER_NUMBER}/${LAYER_TYPE}/${DATE}/"
mkdir -p "${EXP_DIR}/"
LOGFILE="${EXP_DIR}${DATE}-rec-experiments.txt"

mkdir -p "$EXP_DIR"

echo "Running experiments on $DATE" >> "$LOGFILE"

for RADIUS in $(seq 0.0 $FRACTION 1.0); do
    printf -v RAD "%.2f" "$RADIUS"
    echo "Running: -g rn $RAD -e $EPOCHS" | tee -a "$LOGFILE"
    TIMESTAMP=$(date +%y%m%d_%H%M%S)
    FILENAME="${EXP_DIR}${TIMESTAMP}_${DATASET}_ln${LAYER_NUMBER}_rec_${RAD}_sl${SELF_LOOP}_${EPOCHS}.json"
    python -u main.py -d $DATASET -sp $FILENAME -tg rec $RAD -e $EPOCHS \
        -hd $HIDDEN_DIM -ln $LAYER_NUMBER -lr $LR -m $STGI_MODE -sl $SELF_LOOP -bs $BATCH_SIZE -v 0 | tee -a "$LOGFILE"
done
