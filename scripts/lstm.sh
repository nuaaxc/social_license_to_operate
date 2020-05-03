#!/usr/bin/env bash

source include.sh

# REL DATA (train)
#python models/LSTM/trainer.py \
#          --train \
#          --data_dir "$data_dir" \
#          --word_vec_dir "$word_vec_dir" \
#          --dataset REL \
#          --embed_dim 200 \
#          --hidden_dim 256 \
#          --dropout 0.3 \
#          --learning_rate 0.5 \
#          --batch_size 128 \
#          --gpus 1 \
#          --pwe 1

# ----------------------------------------------

# CATE_SOCIAL DATA (train)
#python models/LSTM/trainer.py \
#          --train \
#          --data_dir "$data_dir" \
#          --word_vec_dir "$word_vec_dir" \
#          --dataset CATE_SOCIAL \
#          --embed_dim 200 \
#          --hidden_dim 256 \
#          --dropout 0.3 \
#          --learning_rate 0.5 \
#          --batch_size 128 \
#          --gpus 1 \
#          --pwe 1

# CATE_ECONOMIC DATA (train)
#python models/LSTM/trainer.py \
#          --train \
#          --data_dir "$data_dir" \
#          --word_vec_dir "$word_vec_dir" \
#          --dataset CATE_ECONOMIC \
#          --embed_dim 200 \
#          --hidden_dim 256 \
#          --dropout 0.3 \
#          --learning_rate 0.5 \
#          --batch_size 128 \
#          --gpus 1 \
#          --pwe 1


# CATE_ENVIRONMENTAL DATA (train)
#python models/LSTM/trainer.py \
#          --train \
#          --data_dir "$data_dir" \
#          --word_vec_dir "$word_vec_dir" \
#          --dataset CATE_ENVIRONMENTAL \
#          --embed_dim 200 \
#          --hidden_dim 256 \
#          --dropout 0.3 \
#          --learning_rate 0.5 \
#          --batch_size 128 \
#          --gpus 1 \
#          --pwe 1


# CATE_OTHER DATA (train)
#python models/LSTM/trainer.py \
#          --train \
#          --data_dir "$data_dir" \
#          --word_vec_dir "$word_vec_dir" \
#          --dataset CATE_OTHER \
#          --embed_dim 200 \
#          --hidden_dim 256 \
#          --dropout 0.3 \
#          --learning_rate 0.5 \
#          --batch_size 128 \
#          --gpus 1 \
#          --pwe 1

# ----------------------------------------------

# STANCE_SOCIAL DATA (train)
#python models/LSTM/trainer.py \
#          --train \
#          --data_dir "$data_dir" \
#          --word_vec_dir "$word_vec_dir" \
#          --dataset STANCE_SOCIAL \
#          --embed_dim 200 \
#          --hidden_dim 256 \
#          --dropout 0.3 \
#          --learning_rate 0.5 \
#          --batch_size 128 \
#          --gpus 1 \
#          --pwe 1

# STANCE_ECONOMIC DATA (train)
#python models/LSTM/trainer.py \
#          --train \
#          --data_dir "$data_dir" \
#          --word_vec_dir "$word_vec_dir" \
#          --dataset STANCE_ECONOMIC \
#          --embed_dim 200 \
#          --hidden_dim 256 \
#          --dropout 0.3 \
#          --learning_rate 0.5 \
#          --batch_size 128 \
#          --gpus 1 \
#          --pwe 1


# STANCE_ENVIRONMENTAL DATA (train)
#python models/LSTM/trainer.py \
#          --train \
#          --data_dir "$data_dir" \
#          --word_vec_dir "$word_vec_dir" \
#          --dataset STANCE_ENVIRONMENTAL \
#          --embed_dim 200 \
#          --hidden_dim 256 \
#          --dropout 0.3 \
#          --learning_rate 0.5 \
#          --batch_size 128 \
#          --gpus 1 \
#          --pwe 1


# STANCE_OTHER DATA (train)
#python models/LSTM/trainer.py \
#          --train \
#          --data_dir "$data_dir" \
#          --word_vec_dir "$word_vec_dir" \
#          --dataset STANCE_OTHER \
#          --embed_dim 200 \
#          --hidden_dim 256 \
#          --dropout 0.3 \
#          --learning_rate 0.5 \
#          --batch_size 128 \
#          --gpus 1 \
#          --pwe 1

# ----------------------------------------------

# CATE_STANCE_SOCIAL DATA (train)
python models/LSTM/trainer.py \
          --train \
          --data_dir "$data_dir" \
          --word_vec_dir "$word_vec_dir" \
          --dataset CATE_STANCE_SOCIAL \
          --embed_dim 200 \
          --hidden_dim 256 \
          --dropout 0.3 \
          --learning_rate 0.5 \
          --batch_size 128 \
          --gpus 1 \
          --pwe 1

# CATE_STANCE_ECONOMIC DATA (train)
python models/LSTM/trainer.py \
          --train \
          --data_dir "$data_dir" \
          --word_vec_dir "$word_vec_dir" \
          --dataset CATE_STANCE_ECONOMIC \
          --embed_dim 200 \
          --hidden_dim 256 \
          --dropout 0.3 \
          --learning_rate 0.5 \
          --batch_size 128 \
          --gpus 1 \
          --pwe 1


# CATE_STANCE_ENVIRONMENTAL DATA (train)
python models/LSTM/trainer.py \
          --train \
          --data_dir "$data_dir" \
          --word_vec_dir "$word_vec_dir" \
          --dataset CATE_STANCE_ENVIRONMENTAL \
          --embed_dim 200 \
          --hidden_dim 256 \
          --dropout 0.3 \
          --learning_rate 0.5 \
          --batch_size 128 \
          --gpus 1 \
          --pwe 1


# CATE_STANCE_OTHER DATA (train)
python models/LSTM/trainer.py \
          --train \
          --data_dir "$data_dir" \
          --word_vec_dir "$word_vec_dir" \
          --dataset CATE_STANCE_OTHER \
          --embed_dim 200 \
          --hidden_dim 256 \
          --dropout 0.3 \
          --learning_rate 0.5 \
          --batch_size 128 \
          --gpus 1 \
          --pwe 1

# FULL DATA (predict & save_feature)
#python models/LSTM/trainer.py \
#          --predict \
#          --weight_path "$src_path/logs_LSTM/version_3/checkpoints/_ckpt_epoch_14.ckpt" \
#          --cfg_path "$src_path/logs_LSTM/version_3/meta_tags.csv" \
#          --data_dir "$data_dir" \
#          --word_vec_dir "$word_vec_dir" \
#          --dataset REL \
#          --embed_dim 200 \
#          --hidden_dim 256 \
#          --dropout 0.3 \
#          --learning_rate 0.5 \
#          --batch_size 16 \
#          --gpus 1 \
#          --pwe 1

# Grid Search
#python src/baselines/LSTM/grid_search.py \
#          --dataset REL \
#          --embed_dim 200 \
#          --gpus 1 \
#          --pwe 1