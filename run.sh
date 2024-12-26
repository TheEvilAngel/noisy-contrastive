### sym noise ratio = 40% CIFAR100 GCE 20241211-090055 tmux[0] 58%
python -u main_firsttest.py --data_root ./data --exp_dir ./save --dataset imagenet100 --lr 0.02 --lamb 8.0 --tau 0.05 --r 0.4 --type gce --beta 0.6 --gpu 0

### sym noise ratio = 40% CIFAR100 CE  20241211-081441 tmux[1] 失败
python -u main_firsttest.py --data_root ./data --exp_dir ./save --dataset imagenet100 --lr 0.02 --lamb 130 --tau 0.05 --r 0.4 --gpu 0

### sym noise ratio = 40% CIFAR10 GCE  20241211-081746 tmux[2] 57% tau = 0.4
python -u main_firsttest.py --data_root ./data --exp_dir ./save --dataset imagenet10 --lr 0.02 --lamb 8.0 --tau 0.4 --r 0.4 --type gce --beta 0.6 --gpu 1

### sym noise ratio = 40% CIFAR10 CE 20241211-081929 tmux[3]
python -u main_firsttest.py --data_root ./data --exp_dir ./save --dataset imagenet10 --lr 0.02 --lamb 90 --tau 0.4 --r 0.4 --gpu 1


### asym noise ratio = 40% CIFAR10 CE 20241211-082118 tmux[4]
python -u main_firsttest.py --data_root ./data --exp_dir ./save --dataset imagenet10 --lr 0.04 --lamb 50 --tau 0.4 --r 0.4 --noise_type asym --gpu 5

### asym noise ratio = 40% CIFAR100 CE 20241211-082118 tmux[5]
python -u main_firsttest.py --data_root ./data --exp_dir ./save --dataset imagenet10 --lr 0.04 --lamb 170 --tau 0.05 --r 0.4 --noise_type asym --gpu 5

### sym noise ratio = 40% CIFAR100 GCE 20241211-070946 tmux[6] 学习率不同
python -u main_firsttest.py --data_root ./data --exp_dir ./save --dataset imagenet100 --lr 0.002 --lamb 8.0 --tau 0.05 --r 0.4 --type gce --beta 0.6 --gpu 0

### sym noise ratio = 40% CIFAR100 GCE 20241211-070946 tmux[7] 学习率不同
python -u main_firsttest.py --data_root ./data --exp_dir ./save --dataset imagenet100 --lr 0.0002 --lamb 8.0 --tau 0.05 --r 0.4 --type gce --beta 0.6 --gpu 0


'''============================================第二轮======================================================='''
### sym noise ratio = 40% CIFAR100 GCE 20241214-224014 tmux[0] 看中间结果有没有更好的
python -u main_firsttest.py --data_root ./data --exp_dir ./save --dataset imagenet100 --lr 0.02 --lamb 8.0 --tau 0.05 --r 0.4 --type gce --beta 0.6 --gpu 4

### sym noise ratio = 40% CIFAR100 GCE 20241214-224103 tmux[1] 看输入都是一样的数据好不好
python -u main_secondtest.py --data_root ./data --exp_dir ./save --dataset imagenet100 --lr 0.02 --lamb 8.0 --tau 0.05 --r 0.4 --type gce --beta 0.6 --gpu 4

### sym noise ratio = 40% CIFAR100 GCE 20241214-224014 tmux[2] 改变tau看结果
python -u main_firsttest.py --data_root ./data --exp_dir ./save --dataset imagenet100 --lr 0.02 --lamb 8.0 --tau 0.1 --r 0.4 --type gce --beta 0.6 --gpu 4

### sym noise ratio = 40% CIFAR100 GCE 20241214-224014 tmux[3] 改变tau看结果
python -u main_firsttest.py --data_root ./data --exp_dir ./save --dataset imagenet100 --lr 0.02 --lamb 8.0 --tau 0.15 --r 0.4 --type gce --beta 0.6 --gpu 4

### sym noise ratio = 40% CIFAR100 GCE 20241214-224014 tmux[4] 改变tau看结果
python -u main_firsttest.py --data_root ./data --exp_dir ./save --dataset imagenet100 --lr 0.02 --lamb 8.0 --tau 0.2 --r 0.4 --type gce --beta 0.6 --gpu 4

### sym noise ratio = 40% CIFAR100 GCE 20241214-225000 tmux[5] 改变tau看结果 59%最优秀
python -u main_firsttest.py --data_root ./data --exp_dir ./save --dataset imagenet100 --lr 0.02 --lamb 8.0 --tau 0.25 --r 0.4 --type gce --beta 0.6 --gpu 7

### sym noise ratio = 40% CIFAR100 GCE 20241214-224014 tmux[6] 改变tau看结果
python -u main_firsttest.py --data_root ./data --exp_dir ./save --dataset imagenet100 --lr 0.02 --lamb 8.0 --tau 0.3 --r 0.4 --type gce --beta 0.6 --gpu 7

### sym noise ratio = 40% CIFAR100 GCE 20241214-224014 tmux[7] 改变tau看结果
python -u main_firsttest.py --data_root ./data --exp_dir ./save --dataset imagenet100 --lr 0.02 --lamb 8.0 --tau 0.35 --r 0.4 --type gce --beta 0.6 --gpu 7

'''============================================第三轮======================================================='''

### sym noise ratio = 40% CIFAR100 GCE 20241214-225000 tmux[1] 改变tau看结果，10轮存一次 
python -u main_firsttest.py --data_root ./data --exp_dir ./save --dataset imagenet100 --lr 0.02 --lamb 8.0 --tau 0.25 --r 0.4 --type gce --beta 0.6 --gpu 7

### sym noise ratio = 40% CIFAR100 GCE 20241214-225000 tmux[2] 改变tau看结果，10轮存一次 
python -u main_firsttest.py --data_root ./data --exp_dir ./save --dataset imagenet100 --lr 0.02 --lamb 8.0 --tau 0.24 --r 0.4 --type gce --beta 0.6 --gpu 7

### sym noise ratio = 40% CIFAR100 GCE 20241214-225000 tmux[3] 改变tau看结果，10轮存一次 
python -u main_firsttest.py --data_root ./data --exp_dir ./save --dataset imagenet100 --lr 0.02 --lamb 8.0 --tau 0.26 --r 0.4 --type gce --beta 0.6 --gpu 7

### sym noise ratio = 40% CIFAR100 GCE 20241214-225000 tmux[4] 改变tau看结果，10轮存一次 
python -u main_firsttest.py --data_root ./data --exp_dir ./save --dataset imagenet100 --lr 0.02 --lamb 8.0 --tau 0.27 --r 0.4 --type gce --beta 0.6 --gpu 5

### sym noise ratio = 40% CIFAR100 GCE 20241214-225000 tmux[5] 改变tau看结果，10轮存一次 
python -u main_firsttest.py --data_root ./data --exp_dir ./save --dataset imagenet100 --lr 0.02 --lamb 8.0 --tau 0.23 --r 0.4 --type gce --beta 0.6 --gpu 5

### sym noise ratio = 40% CIFAR100 GCE 20241214-224103 tmux[6] 修改了loss函数，和论文一致
python -u main_thirdtest.py --data_root ./data --exp_dir ./save --dataset imagenet100 --lr 0.02 --lamb 8.0 --tau 0.05 --r 0.4 --type gce --beta 0.6 --gpu 6

### sym noise ratio = 40% CIFAR100 GCE 20241214-224103 tmux[7] 修改了loss函数，和论文一致
python -u main_thirdtest.py --data_root ./data --exp_dir ./save --dataset imagenet100 --lr 0.02 --lamb 8.0 --tau 0.25 --r 0.4 --type gce --beta 0.6 --gpu 6



### sym noise ratio = 40% CIFAR-100
python -u main.py --data_root ./data --dataset cifar100  --lr 0.02 --lamb 8.0 --tau 0.05 --r 0.4 --noise_type sym --type gce --beta 0.6 --gpu 0 --epochs 2

################
# CIFAR-10: CE #
################

### noise ratio = 0%
python -u main.py --data_root ./data --dataset cifar10 --lr 0.02 --lamb 50 --tau 0.8 --r 0 --noise_type sym --gpu 4 --epochs 2

### sym noise ratio = 20%
python -u main.py --data_root /data --dataset cifar10  --lr 0.02 --lamb 50 -- tau 0.4 --r 0.2 --noise_type sym

### sym noise ratio = 40%
python -u main.py --data_root ./data --dataset cifar10  --lr 0.02 --lamb 90 --tau 0.4 --r 0.4 --noise_type sym --gpu 3

### sym noise ratio = 60%
python -u main.py --data_root /data --dataset cifar10  --lr 0.02 --lamb 170 -- tau 0.8 --r 0.6 --noise_type sym

### sym noise ratio = 80%
python -u main.py --data_root /data --dataset cifar10  --lr 0.02 --lamb 130 -- tau 0.4 --r 0.8 --noise_type sym

### sym noise ratio = 90%
python -u main.py --data_root /data --dataset cifar10  --lr 0.02 --lamb 170 -- tau 0.4 --r 0.9 --noise_type sym

### asym noise ratio = 40%
python -u main.py --data_root /data --dataset cifar10  --lr 0.04 --lamb 50 -- tau 0.4 --r 0.4 --noise_type asym



################
# CIFAR-100: CE #
################

### noise ratio = 0%
python -u main.py --data_root /data --dataset cifar100 --lr 0.02 --lamb 50 -- tau 0.05 --r 0 --noise_type sym

### sym noise ratio = 20%
python -u main.py --data_root /data --dataset cifar100  --lr 0.02 --lamb 130 -- tau 0.05 --r 0.2 --noise_type sym

### sym noise ratio = 40%
python -u main.py --data_root /data --dataset cifar100  --lr 0.02 --lamb 130 -- tau 0.05 --r 0.4 --noise_type sym

### sym noise ratio = 60%
python -u main.py --data_root /data --dataset cifar100  --lr 0.02 --lamb 130 -- tau 0.05 --r 0.6 --noise_type sym

### sym noise ratio = 80%
python -u main.py --data_root /data --dataset cifar100  --lr 0.02 --lamb 130 -- tau 0.05 --r 0.8 --noise_type sym

### asym noise ratio = 40%
python -u main.py --data_root /data --dataset cifar100  --lr 0.04 --lamb 170 -- tau 0.05 --r 0.4 --noise_type asym


################
# CIFAR-10: GCE # Since GCE is partial noise-robust, lamb can be set much smaller
################

### sym noise ratio = 20%
python -u main.py --data_root /data --dataset cifar10  --lr 0.02 --lamb 4.0 -- tau 0.4 --r 0.2 --noise_type sym --type gce --beta 0.6

### sym noise ratio = 40%
python -u main.py --data_root /data --dataset cifar10  --lr 0.02 --lamb 8.0 -- tau 0.4 --r 0.4 --noise_type sym --type gce --beta 0.6

### sym noise ratio = 60%
python -u main.py --data_root /data --dataset cifar10  --lr 0.02 --lamb 4.0 -- tau 0.8 --r 0.6 --noise_type sym --type gce --beta 0.8

### sym noise ratio = 80%
python -u main.py --data_root /data --dataset cifar10  --lr 0.02 --lamb 8.0 -- tau 0.8 --r 0.8 --noise_type sym --type gce --beta 0.8

### sym noise ratio = 90%
python -u main.py --data_root /data --dataset cifar10  --lr 0.02 --lamb 8.0 -- tau 0.4 --r 0.9 --noise_type sym --type gce --beta 0.8


################
# CIFAR-100: GCE #
################

### noise ratio = 0%
python -u main.py --data_root /data --dataset cifar100 --lr 0.02 --lamb 8.0 -- tau 0.05 --r 0 --noise_type sym --type gce --beta 0.6

### sym noise ratio = 20%
python -u main.py --data_root /data --dataset cifar100  --lr 0.02 --lamb 8.0 -- tau 0.05 --r 0.2 --noise_type sym --type gce --beta 0.6

### sym noise ratio = 40%
python -u main.py --data_root /data --dataset cifar100  --lr 0.02 --lamb 8.0 -- tau 0.05 --r 0.4 --noise_type sym --type gce --beta 0.6

### sym noise ratio = 60%
python -u main.py --data_root /data --dataset cifar100  --lr 0.02 --lamb 8.0 -- tau 0.05 --r 0.6 --noise_type sym --type gce --beta 0.6

### sym noise ratio = 80%
python -u main.py --data_root /data --dataset cifar100  --lr 0.02 --lamb 8.0 -- tau 0.05 --r 0.8 --noise_type sym --type gce --beta 0.6
