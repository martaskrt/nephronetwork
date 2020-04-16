#!/bash/bin

python3 train_vae.py --epochs 15 --model_dir vae_model --beta 0.01 --lr 0.0002 --exp beta=0.01,lr=0.0002
python3 train_vae.py --epochs 15 --model_dir vae_model --beta 0.1 --lr 0.0002 --exp beta=0.1,lr=0.0002
python3 train_vae.py --epochs 15 --model_dir vae_model --beta 1 --lr 0.0002 --exp beta=1,lr=0.0002
python3 train_vae.py --epochs 15 --model_dir vae_model --beta 10 --lr 0.0002 --exp beta=10,lr=0.0002
python3 train_vae.py --epochs 15 --model_dir vae_model --beta 100 --lr 0.0002 --exp beta=100,lr=0.0002
