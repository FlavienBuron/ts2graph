#!/bin/bash

. .venv/bin/activate

EPOCHS=50
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
EXP_DIR="./experiments/results/knn/"
LOGFILE="${EXP_DIR}${DATE}-knn-experiments.txt"

mkdir -p "$EXP_DIR"

echo "Running experiments on $DATE" >> "$LOGFILE"

declare -A TECHNIQUES=(
    ["zero_0"]=0
    ["zero_1"]=1
    ["one_1"]=1
    ["one_0"]=0
    ["loc"]=0.5
)

USE_TEMP=""
if [ "$USE_TEMPORAL" -eq 1 ]; then
    USE_TEMP="-ut"
fi

ORIGINAL=$SELF_LOOP

# Loop through fixed graphs techniques
for G in "${!TECHNIQUES[@]}"; do
    V=${TECHNIQUES[$G]}

        # Reset default self-loop
        SELF_LOOP=$ORIGINAL
        BASE_G=$G

        # Check if technique is a variant of zero or one
        if [[ "$G" == zero_* ]]; then
            BASE_G="zero"
            SELF_LOOP=${G#zero_}
        elif [[ "$G" == one_* ]]; then
            BASE_G="one"
            SELF_LOOP=${G#one_}
        fi

    echo "Running: -g $G $V -e $EPOCHS" | tee -a "$LOGFILE"
    TIMESTAMP=$(date +%y%m%d_%H%M%S)
    FILENAME="${EXP_DIR}${TIMESTAMP}_${DATASET}_ln${LAYER_NUMBER}_${G}_${V}_sl${SELF_LOOP}_${EPOCHS}.json"
    python -u main.py -d $DATASET -sp $FILENAME -g "$G" "$V" -e "$EPOCHS" -hd $HIDDEN_DIM -ln $LAYER_NUMBER -lr $LR $USE_TEMP -sl $SELF_LOOP -v 0 | tee -a "$LOGFILE"
done

SELF_LOOP=$ORIGINAL
# Sweep knn values from 1 to KNN_MAX
KNN_MAX=$(awk -v n=$NUM_NODES -v f=$FRACTION 'BEGIN { printf "%d", (n * f + 0.5) }')
for ((K=1; K<=KNN_MAX; K++)); do
    echo "Running: -g knn $K -e $EPOCHS" | tee -a "$LOGFILE"
    TIMESTAMP=$(date +%y%m%d_%H%M%S)
    FILENAME="${EXP_DIR}${TIMESTAMP}_${DATASET}_ln${LAYER_NUMBER}_knn_${K}_sl${SELF_LOOP}_${EPOCHS}.json"
    python -u main.py -d $DATASET -sp $FILENAME -g knn $K -e $EPOCHS \
           -hd $HIDDEN_DIM -ln $LAYER_NUMBER -lr $LR $USE_TEMP -sl $SELF_LOOP -v 0 | tee -a "$LOGFILE"
done
