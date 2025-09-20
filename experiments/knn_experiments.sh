#!/bin/bash

. .venv/bin/activate

EPOCHS=30
BATCH_SIZE=128
HIDDEN_DIM=32
LAYER_NUMBER=1
SELF_LOOP=0
STGI_MODE='s'
DATASET="airq_small"
NUM_NODES=36
FRACTION=0.05
LAYER_TYPE="GCNConv"
FULL_DATASET=0
MODEL="stgi"
SHUFFLE=0


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
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --shuffle)
            SHUFFLE=1
            shift
            ;;
        --model)
            MODEL="$2"
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
        --full_dataset)
            FULL_DATASET=1
            shift
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

EXP_DIR="./experiments/results/${MODEL}/knn/ln${LAYER_NUMBER}/${LAYER_TYPE}/${DATE}/"
mkdir -p "${EXP_DIR}/"
LOGFILE="${EXP_DIR}${DATE}-knn-experiments.txt"

echo "Running experiments on $DATE" >> "$LOGFILE"

declare -A TECHNIQUES=(
    ["zero_0"]=0
    ["one_0"]=0
    ["loc"]=0.5
)

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

    echo "Running: $MODEL -g $G $V -e $EPOCHS" | tee -a "$LOGFILE"
    TIMESTAMP=$(date +%y%m%d_%H%M%S)
    FILENAME="${EXP_DIR}${TIMESTAMP}_${MODEL}_${DATASET}_${STGI_MODE}_ln${LAYER_NUMBER}_${BASE_G}_${V}_sl${SELF_LOOP}_${EPOCHS}_${BATCH_SIZE}_${SHUFFLE}.json"
    python -u main.py --model $MODEL -d $DATASET -bs $BATCH_SIZE -sp $FILENAME -sg "$BASE_G" "$V" -e "$EPOCHS" -l $LAYER_TYPE -hd $HIDDEN_DIM -ln $LAYER_NUMBER -lr $LR -m $STGI_MODE -sl $SELF_LOOP \
        $( [[ "$FULL_DATASET" -eq 1 ]] && echo -fd ) \
        $( [[ "$SHUFFLE" -eq 1 ]] && echo -sb ) \
        -v 0 | tee -a "$LOGFILE"
done

SELF_LOOP=$ORIGINAL

for K in $(seq 0.0 $FRACTION 1.0); do
    echo "Running: $MODEL -g knn $K -e $EPOCHS -l $LAYER_TYPE" | tee -a "$LOGFILE"
    TIMESTAMP=$(date +%y%m%d_%H%M%S)
    FILENAME="${EXP_DIR}${TIMESTAMP}_${MODEL}_${DATASET}_${STGI_MODE}_ln${LAYER_NUMBER}_knn_${K}_sl${SELF_LOOP}_${EPOCHS}_${BATCH_SIZE}_${SHUFFLE}.json"
    python -u main.py --model $MODEL -d $DATASET -bs $BATCH_SIZE -sp $FILENAME -sg knn $K -e $EPOCHS -l $LAYER_TYPE \
           -hd $HIDDEN_DIM -ln $LAYER_NUMBER -lr $LR -m $STGI_MODE -sl $SELF_LOOP \
        $( [[ "$FULL_DATASET" -eq 1 ]] && echo -fd ) \
        $( [[ "$SHUFFLE" -eq 1 ]] && echo -sb ) \
         -v 0 | tee -a "$LOGFILE"
done
