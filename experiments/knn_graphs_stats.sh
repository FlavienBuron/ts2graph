.#!/bin/bash

. .venv/bin/activate

# Accept custom list of epochs from command line, or use defaults
FRAC=("$@")
if [ ${#FRAC[@]} -eq 0 ]; then
    FRAC=({1..100})
fi

HIDDEN_DIM=32
LAYER_NUMBER=1
SELF_LOOP=0
USE_MLP_OUTPUT=0
MLP_SIZE=32
DATASET="airq_small"
TOTAL=36

for K in {1..10} 12 14 16 18 20 22 24 26 28 30 32 34 36; do
    echo "Running: KNN $K for $DATASET"
    python -u main.py -d $DATASET -g knn "$K" -e 1 -hd $HIDDEN_DIM -ln $LAYER_NUMBER -gs -dt -v 0
done

# for F in "${FRAC[@]}"; do
#     K=$(awk "BEGIN { printf \"%d\", ($TOTAL * $F) / 100 }")
#     echo "Running: KNN $K for $DATASET"
#     python -u main.py -d $DATASET -g knn "$K" -e 1 -hd $HIDDEN_DIM -ln $LAYER_NUMBER -gs -dt -v 0
# done
