#!/bin/bash

. .venv/bin/activate

EPOCHS=30
UNWEIGHTED=0
HIDDEN_DIM=32
LAYER_NUMBER=1
SELF_LOOP=0
STGI_MODE='s'
MLP_SIZE=32
DATASET="airq_small"
NUM_NODES=36
FRACTION=0.05
LAYER_TYPE="GCNConv"

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
        --unweighted)
            UNWEIGHTED=1
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
        LR=0.005
    else
        LR=0.0004
    fi
fi


DATE=$(date +%y%m%d)
EXP_DIR="./experiments/results/loc/ln${LAYER_NUMBER}/${LAYER_TYPE}/"
mkdir -p "$EXP_DIR"
LOGFILE="${EXP_DIR}${DATE}-loc-experiments.txt"

echo "Running experiments on $DATE" >> "$LOGFILE"

KNN_VAL=50
[ "$DATASET" == "airq_small" ] && KNN_VAL=3

declare -A TECHNIQUES=(
    ["zero_0"]=0
    # ["zero_1"]=1
    # ["one_1"]=1
    ["one_0"]=0
    ["knn"]=$KNN_VAL
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

    echo "Running: -g $BASE_G $V -e $EPOCHS" | tee -a "$LOGFILE"
    TIMESTAMP=$(date +%y%m%d_%H%M%S)
    FILENAME="${EXP_DIR}${TIMESTAMP}_${DATASET}_${STGI_MODE}_ln${LAYER_NUMBER}_${BASE_G}_${V}_sl${SELF_LOOP}_${EPOCHS}.json"
    python -u main.py -d $DATASET -sp $FILENAME -sg "$BASE_G" "$V" -e "$EPOCHS" -l $LAYER_TYPE -hd $HIDDEN_DIM -ln $LAYER_NUMBER -lr $LR -m $STGI_MODE -sl $SELF_LOOP \
        $( [ "$UNWEIGHTED" -eq 1 ] && echo -ug ) \
        -v 0 | tee -a "$LOGFILE"
done

SELF_LOOP=$ORIGINAL

# Sweep knn values from 1 to KNN_MAX
for LOC in $(seq 0.0 $FRACTION 1.0); do
    printf -v LOC_FMT "%.2f" "$LOC"
    echo "Running: -g loc $LOC_FMT -e $EPOCHS -l $LAYER_TYPE" | tee -a "$LOGFILE"
    TIMESTAMP=$(date +%y%m%d_%H%M%S)
    FILENAME="${EXP_DIR}${TIMESTAMP}_${DATASET}_${STGI_MODE}_ln${LAYER_NUMBER}_loc_${LOC_FMT}_sl${SELF_LOOP}_${EPOCHS}.json"
    python -u main.py -d $DATASET -sp $FILENAME -sg loc $LOC_FMT -e $EPOCHS -l $LAYER_TYPE \
           -hd $HIDDEN_DIM -ln $LAYER_NUMBER -lr $LR -m $STGI_MODE -sl $SELF_LOOP \
        $( [ "$UNWEIGHTED" -eq 1 ] && echo -ug ) \
        -v 0 | tee -a "$LOGFILE"
done
