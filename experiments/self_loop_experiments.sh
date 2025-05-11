    . .venv/bin/activate

    # Get current date in YYMMDD format
    DATE=$(date +%y%m%d)
    LOGFILE="./experiments/results/${DATE}-sl-experiments.txt"

    echo "Running experiments on $DATE" >> "$LOGFILE"

    # Test the impact of self loops

    # 1. Defaults OFF
    python -u main.py -d air -g zero 0 -e 10 -hd 32 -ln 1 -v 0 | tee -a "$LOGFILE"
    python -u main.py -d air -g one 1 -e 10 -hd 32 -ln 1 -v 0 | tee -a "$LOGFILE"
    python -u main.py -d air -g loc 0.5 -e 10 -hd 32 -ln 1 -v 0 | tee -a "$LOGFILE"
    python -u main.py -d air -g knn 50 -e 10 -hd 32 -ln 1 -v 0 | tee -a "$LOGFILE"

    # 2. ON
    python -u main.py -d air -g zero 0 -e 10 -hd 32 -sl 1 -ln 1 -v 0 | tee -a "$LOGFILE"
    python -u main.py -d air -g one 1 -e 10 -hd 32 -sl 1 -ln 1 -v 0 | tee -a "$LOGFILE"
    python -u main.py -d air -g loc 0.5 -e 10 -hd 32 -sl 1 -ln 1 -v 0 | tee -a "$LOGFILE"
    python -u main.py -d air -g knn 50 -e 10 -hd 32 -sl 1 -ln 1 -v 0 | tee -a "$LOGFILE"    
