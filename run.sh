### sym noise ratio = 40%
python -u main_firsttest.py --data_root ./data --exp_dir ./save --dataset imagenet --lr 0.02 --lamb 8.0 --tau 0.05 --r 0.4 --type gce --beta 0.6 --gpu 0

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
