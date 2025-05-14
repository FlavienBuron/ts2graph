    . .venv/bin/activate

    # Get current date in YYMMDD format
    DATE=$(date +%y%m%d)
    LOGFILE="./experiments/results/${DATE}-e-experiments.txt"

    echo "Running experiments on $DATE" >> "$LOGFILE"

    # Test the impact of number of epochs

    # 1. Defaults e=10
    python -u main.py -d air -g zero 0 -e 10 -hd 32 -ln 1 -v 0 | tee -a "$LOGFILE"
    python -u main.py -d air -g one 1 -e 10 -hd 32 -ln 1 -v 0 | tee -a "$LOGFILE"
    python -u main.py -d air -g loc 0.5 -e 10 -hd 32 -ln 1 -v 0 | tee -a "$LOGFILE"
    python -u main.py -d air -g knn 50 -e 10 -hd 32 -ln 1 -v 0 | tee -a "$LOGFILE"

    # 2. e=5
    python -u main.py -d air -g zero 0 -e 5 -hd 32 -ln 1 -v 0 | tee -a "$LOGFILE"
    python -u main.py -d air -g one 1 -e 5 -hd 32 -ln 1 -v 0 | tee -a "$LOGFILE"
    python -u main.py -d air -g loc 0.5 -e 5 -hd 32 -ln 1 -v 0 | tee -a "$LOGFILE"
    python -u main.py -d air -g knn 50 -e 5 -hd 32 -ln 1 -v 0 | tee -a "$LOGFILE"    

    # 3. e=15
    python -u main.py -d air -g zero 0 -e 15 -hd 32 -ln 1 -v 0 | tee -a "$LOGFILE"
    python -u main.py -d air -g one 1 -e 15 -hd 32 -ln 1 -v 0 | tee -a "$LOGFILE"
    python -u main.py -d air -g loc 0.5 -e 15 -hd 32 -ln 1 -v 0 | tee -a "$LOGFILE"
    python -u main.py -d air -g knn 50 -e 15 -hd 32 -ln 1 -v 0 | tee -a "$LOGFILE"    

    # 4. e=20
    python -u main.py -d air -g zero 0 -e 20 -hd 32 -ln 1 -v 0 | tee -a "$LOGFILE"
    python -u main.py -d air -g one 1 -e 20 -hd 32 -ln 1 -v 0 | tee -a "$LOGFILE"
    python -u main.py -d air -g loc 0.5 -e 20 -hd 32 -ln 1 -v 0 | tee -a "$LOGFILE"
    python -u main.py -d air -g knn 50 -e 20 -hd 32 -ln 1 -v 0 | tee -a "$LOGFILE"    
