    . .venv/bin/activate

    # Get current date in YYMMDD format
    DATE=$(date +%y%m%d)
    LOGFILE="./experiments/results/${DATE}-lr-experiments.txt"

    echo "Running experiments on $DATE" >> "$LOGFILE"

    # Test the impact of learning rate

    # 1. Defaults lr=1e-3
    python -u main.py -d air -g zero 0 -e 10 -hd 32 -ln 1 -v 0 | tee -a "$LOGFILE"
    python -u main.py -d air -g one 1 -e 10 -hd 32 -ln 1 -v 0 | tee -a "$LOGFILE"
    python -u main.py -d air -g loc 0.5 -e 10 -hd 32 -ln 1 -v 0 | tee -a "$LOGFILE"
    python -u main.py -d air -g knn 100 -e 10 -hd 32 -ln 1 -v 0 | tee -a "$LOGFILE"

    # 2. 1e-2
    python -u main.py -d air -g zero 0 -e 10 -hd 32 -lr 0.01 -ln 1 -v 0 | tee -a "$LOGFILE"
    python -u main.py -d air -g one 1 -e 10 -hd 32 -lr 0.01 -ln 1 -v 0 | tee -a "$LOGFILE"
    python -u main.py -d air -g loc 0.5 -e 10 -hd 32 -lr 0.01 -ln 1 -v 0 | tee -a "$LOGFILE"
    python -u main.py -d air -g knn 100 -e 10 -hd 32 -lr 0.01 -ln 1 -v 0 | tee -a "$LOGFILE"

    # 3. 1e-4 
    python -u main.py -d air -g zero 0 -e 10 -hd 32 -lr 0.0001 -ln 1 -v 0 | tee -a "$LOGFILE"
    python -u main.py -d air -g one 1 -e 10 -hd 32 -lr 0.0001 -ln 1 -v 0 | tee -a "$LOGFILE"
    python -u main.py -d air -g loc 0.5 -e 10 -hd 32 -lr 0.0001 -ln 1 -v 0 | tee -a "$LOGFILE"
    python -u main.py -d air -g knn 100 -e 10 -hd 32 -lr 0.0001 -ln 1 -v 0 | tee -a "$LOGFILE"
    
