source .venv/bin/activate
# Test the impact of hidden layer size
  # 1. Defaults 
python main.py -d air -g zero 0 -i 1 -e 10 -hd 32 -o 16
python main.py -d air -g one 1 -i 1 -e 10 -hd 32 -o 16
python main.py -d air -g loc 0.5 -i 1 -e 10 -hd 32 -o 16
python main.py -d air -g knn 100 -i 1 -e 10 -hd 32 -o 16

  # 2. Size 8 
python main.py -d air -g zero 0 -i 1 -e 10 -hd 8 -o 16
python main.py -d air -g one 1 -i 1 -e 10 -hd 8 -o 16
python main.py -d air -g loc 0.5 -i 1 -e 10 -hd 8 -o 16
python main.py -d air -g knn 100 -i 1 -e 10 -hd 8 -o 16
  # 3. size 16
python main.py -d air -g zero 0 -i 1 -e 10 -hd 16 -o 16
python main.py -d air -g one 1 -i 1 -e 10 -hd 16 -o 16
python main.py -d air -g loc 0.5 -i 1 -e 10 -hd 16 -o 16
python main.py -d air -g knn 100 -i 1 -e 10 -hd 16 -o 16
  # 4. size 64
python main.py -d air -g zero 0 -i 1 -e 10 -hd 64 -o 16
python main.py -d air -g one 1 -i 1 -e 10 -hd 64 -o 16
python main.py -d air -g loc 0.5 -i 1 -e 10 -hd 64 -o 16
python main.py -d air -g knn 100 -i 1 -e 10 -hd 64 -o 16
