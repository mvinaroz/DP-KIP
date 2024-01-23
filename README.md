# DP-KIP
## Installation JAX environment
      conda create --name dpkip python=3.9
      conda activate dpkip
      pip install -e /path/to/the/code/folder/of/this/repo/on/your/local/drive

Note: This will install JAX for CPU only. If you want to install JAX for GPU please go to: [https://github.com/google/jax#installation](https://github.com/google/jax#installation)

## Run KIP and DP-KIP code

### Image data KRR downstream classifier for infinite-width FC-NTK

 `python dpkip_inf_ntk.py --dpsgd=True --l2_norm_clip=1e-6 --epochs=10 --learning_rate=1e-2 --batch_size=50 --epsilon=1 --architecture='FC' --width=1024 --dataset='mnist' --support_size=10` 
 
### Image data KRR downstream classifier for ScatterNet features

`python dp_kip_other_features.py --dpsgd=True --learning_rate=1e-1 --batch_size=2000 --kip_loss_reg=1e-3 --feature_type="wavelet" --dataset='mnist' --epochs=10 --rand_init=True --support_size=10 --l2_norm_clip=1e-2 --epsilon=10`

### Image data KRR downstream classifier for ScatterNet features (non-dp)

`python dp_kip_other_features.py --dpsgd=False --learning_rate=1e-1 --batch_size=2000 --kip_loss_reg=1e-3 --feature_type="wavelet" --dataset='mnist' --epochs=10 --support_size=10`

### Image data KRR downstream classifier for PFs (non-dp)

`python dp_kip_other_features.py --dpsgd=False --learning_rate=1e-1 --batch_size=2000 --kip_loss_reg=1e-3 --feature_type="resnet" --pretrained_encoder=False --normalize_features=True --dataset='svhn_cropped' --epochs=10 --support_size=10`

### Image data KRR downstream classifier for e-NTK (non-dp)

`python KIP_lenet_ntk.py --disable-dp --batch-size 2000 --epochs 10 --lr 1e-1 --reg 1e-3 --sup_size 10 --dataset 'fashion_mnist'`

### Tabular data

 `python dpkip_tab_data.py --dpsgd=True --reg=1e-6 --learning_rate=1e-1 --l2_norm_clip=1e-1 --batch_rate=0.01 --epochs=10 --dataset='credit' --undersampled_rate=0.01 --architecture='FC' --support_size=2 --width=1024`
