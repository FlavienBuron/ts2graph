    . .venv/bin/activate

    # Get current date in YYMMDD format
    DATE=$(date +%y%m%d)
    LOGFILE="./experiments/results/${DATE}-simple-experiments.txt"
    COMMIT_MSG=$(git log -1 --pretty=%B)

    echo "Running experiments on $DATE" >> "$LOGFILE"
    echo "Last commit: $COMMIT_MSG" >> "$LOGFILE"

    # Test the impact of hidden layer size

    # 1. Defaults 
    python -u main.py -d air -g zero 0 -e 10 -hd 32 -v 0 | tee -a "$LOGFILE"
    python -u main.py -d air -g one 1 -e 10 -hd 32 -v 0 | tee -a "$LOGFILE"
    python -u main.py -d air -g loc 0.5 -e 10 -hd 32 -v 0 | tee -a "$LOGFILE"
    python -u main.py -d air -g knn 50 -e 10 -hd 32 -v 0 | tee -a "$LOGFILE"

