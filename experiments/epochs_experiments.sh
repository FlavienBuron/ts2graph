#!/bin/bash

. .venv/bin/activate

EPOCHS=()
BATCH_SIZE=128
HIDDEN_DIM=32
LAYER_NUMBER=1
SELF_LOOP=0
STGI_MODE='s'
MLP_SIZE=32
DATASET="airq_small"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --epochs)
            IFS=',' read -r -a EPOCHS <<< "$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
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

# Accept custom list of epochs from command line, or use defaults
if [ ${#EPOCHS[@]} -eq 0 ]; then
    EPOCHS=(5 10 15 20)
fi

if [[ -z "$LR" || "$LR" == "0" ]]; then
    if [[ "$LAYER_NUMBER" -eq 1 ]]; then
        LR=0.005
    else
        LR=0.0005
    fi
fi


DATE=$(date +%y%m%d)
EXP_DIR="./experiments/results/epochs/"
LOGFILE="${EXP_DIR}${DATE}-e-experiments.txt"

mkdir -p "$EXP_DIR"

echo "Running experiments on $DATE" >> "$LOGFILE"

KNN_VAL=50
[ "$DATASET" == "airq_small" ] && KNN_VAL=3

declare -A TECHNIQUES=(
    ["zero_0"]=0
    ["zero_1"]=1
    ["one_1"]=1
    ["one_0"]=0
    ["loc"]=0.5
    ["knn"]=$KNN_VAL
)

# Loop through epochs and groups
for E in "${EPOCHS[@]}"; do
    for G in "${!TECHNIQUES[@]}"; do
        V=${TECHNIQUES[$G]}

        # Reset default self-loop
        SELF_LOOP=0
        BASE_G=$G

        # Check if technique is a variant of zero or one
        if [[ "$G" == zero_* ]]; then
            BASE_G="zero"
            SELF_LOOP=${G#zero_}
            V=$SELF_LOOP
        elif [[ "$G" == one_* ]]; then
            BASE_G="one"
            SELF_LOOP=${G#one_}
            V=$SELF_LOOP
        fi

        echo "Running: -m $STGI_MODE -g $BASE_G $V -e $E -bs $BATCH_SIZE" | tee -a "$LOGFILE"
        TIMESTAMP=$(date +%y%m%d_%H%M%S)
        FILENAME="${EXP_DIR}${TIMESTAMP}_${DATASET}_${STGI_MODE}_bs{$BATCH_SIZE}_ln${LAYER_NUMBER}_${BASE_G}_${V}_${E}.json"
        python -u main.py -d $DATASET -sp $FILENAME -sg "$BASE_G" "$V" -e "$E" -bs $BATCH_SIZE -hd $HIDDEN_DIM -ln $LAYER_NUMBER -lr $LR -m $STGI_MODE -sl $SELF_LOOP -v 0 | tee -a "$LOGFILE"
    done
done
