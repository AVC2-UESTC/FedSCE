nohup python -u main.py -t 1 -ab 1 -lr 0.01 -jr 1 -lbs 64 -ls 5 -nc 4 -nb 10 -data Office_10 -m Office_10 -fd 512 -did 3 -algo Fedsce -rank 3 -gap 5 -lam 5 -l_i 5  > Office_10-F-jr=1-fd=512-lam=5-sim=C-Fedsce-r3g5-cl=1-gm=1-li=5.out 2>&1 &
wait
echo "Task 1 completed!"
