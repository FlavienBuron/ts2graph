#!/bin/bash

. .venv/bin/activate

EPOCHS=50
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
        --spatial_tech)
            IFS="," read -r -a CUSTOM_SPATIAL <<< "$2"
            shift 2
            ;;
        --tempo_tech)
            IFS="," read -r -a CUSTOM_TEMPO <<< "$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [[ -z "$LR" || "$LR" == "0" ]]; then
    if [[ "$LAYER_NUMBER" -eq 1 ]]; then
        LR=0.005
    else
        LR=0.0005
    fi
fi


DATE=$(date +%y%m%d)
EXP_DIR="./experiments/results/simple/"
LOGFILE="${EXP_DIR}${DATE}-experiments.txt"

mkdir -p "$EXP_DIR"

echo "Running experiments on $DATE" >> "$LOGFILE"

KNN_VAL=50
[ "$DATASET" == "airq_small" ] && KNN_VAL=3

declare -A SPATIAL_TECH
if [ ${#CUSTOM_SPATIAL[@]} -gt 0 ]; then
    for kv in "${CUSTOM_SPATIAL[@]}"; do
        key="${kv%%=*}"
        val="${kv#*=}"
        SPATIAL_TECH[$key]=$val
    done
else SPATIAL_TECH=(
    ["zero_0"]=0
    ["zero_1"]=1
    ["one_1"]=1
    ["one_0"]=0
    ["loc"]=0.5
    ["radius"]=0.5
    ["knn"]=$KNN_VAL
    )
fi

declare -A TEMPO_TECH
if [ ${#CUSTOM_TEMPO[@]} -gt 0 ]; then
    for kv in "${CUSTOM_TEMPO[@]}"; do
        key="${kv%%=*}"
        val="${kv#*=}"
        TEMPO_TECH[$key]=$val
    done
else TEMPO_TECH=(
    ["naive_0"]=0
    ["naive_1"]=1
    ["naive_2"]=2
    ["vis_0"]=0
    ["vis_1"]=1
    )
fi

# Loop through epochs and groups
if [[ "$STGI_MODE" == 's' ]]; then
    for E in "${EPOCHS[@]}"; do
        for G in "${!SPATIAL_TECH[@]}"; do
            V=${SPATIAL_TECH[$G]}

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
            FILENAME="${EXP_DIR}${TIMESTAMP}_${DATASET}_${STGI_MODE}_bs${BATCH_SIZE}_ln${LAYER_NUMBER}_${BASE_G}_${V}_${E}.json"
            python -u main.py -d $DATASET -sp $FILENAME -sg "$BASE_G" "$V" -e "$E" -bs $BATCH_SIZE -hd $HIDDEN_DIM -ln $LAYER_NUMBER -lr $LR -m $STGI_MODE -sl $SELF_LOOP -v 0 | tee -a "$LOGFILE"
        done
    done
elif [[ "$STGI_MODE" == 't' ]]; then
    for E in "${EPOCHS[@]}"; do
        for G in "${!TEMPO_TECH[@]}"; do
            V=${TEMPO_TECH[$G]}
            BASE_G="${G%%_*}"
            IDX="${G#*_}"
            PARAM=$V
            if [[ "$BASE_G" == "vis" ]]; then
                if [[ "$IDX" -eq 0 ]]; then
                    PARAM="nvg"
                else
                    PARAM="hvg"
                fi
            fi

            echo "Running: -m $STGI_MODE -g $BASE_G $V -e $E -bs $BATCH_SIZE" | tee -a "$LOGFILE"
            TIMESTAMP=$(date +%y%m%d_%H%M%S)
            FILENAME="${EXP_DIR}${TIMESTAMP}_${DATASET}_${STGI_MODE}_bs${BATCH_SIZE}_ln${LAYER_NUMBER}_${BASE_G}_${PARAM}_${E}.json"
            python -u main.py -d $DATASET -sp $FILENAME -tg "$BASE_G" "$V" -e "$E" -bs $BATCH_SIZE -hd $HIDDEN_DIM -ln $LAYER_NUMBER -lr $LR -m $STGI_MODE -sl $SELF_LOOP -v 0 | tee -a "$LOGFILE"
        done
    done
else
    for E in "${EPOCHS[@]}"; do
        for G in "${!SPATIAL_TECH[@]}"; do
            V=${SPATIAL_TECH[$G]}

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

            for TG in "${!TEMPO_TECH[@]}"; do
                TG_V=${TEMPO_TECH[$TG]}
                BASE_TG="${TG%%_*}"
                IDX="${TG#*_}"
                PARAM=$TG_V
                if [[ "$BASE_TG" == "vis" ]]; then
                    if [[ "$IDX" -eq 0 ]]; then
                        PARAM="nvg"
                    else
                        PARAM="hvg"
                    fi
                fi

                echo "Running: -m $STGI_MODE -sg $BASE_G $V -tg $BASE_TG $TG_V -e $E -bs $BATCH_SIZE" | tee -a "$LOGFILE"
                TIMESTAMP=$(date +%y%m%d_%H%M%S)
                FILENAME="${EXP_DIR}${TIMESTAMP}_${DATASET}_${STGI_MODE}_bs${BATCH_SIZE}_ln${LAYER_NUMBER}_${BASE_G}_${V}_${BASE_TG}_${PARAM}_${E}.json"
                python -u main.py -d $DATASET -sp $FILENAME -sg "$BASE_G" "$V" -tg "$BASE_TG" "$TG_V" -e "$E" -bs $BATCH_SIZE -hd $HIDDEN_DIM -ln $LAYER_NUMBER -lr $LR -m $STGI_MODE -sl $SELF_LOOP -v 0 | tee -a "$LOGFILE"
            done
        done
    done
fi
