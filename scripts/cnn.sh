#!/usr/bin/env bash

source include.sh

# FULL DATA
#python models/CNN/trainer.py \
#          --train \
#          --data_dir "$data_dir" \
#          --word_vec_dir "$word_vec_dir" \
#          --dataset REL \
#          --embed_dim 200 \
#          --out_channels 256 \
#          --kernel_heights '2, 3, 4, 5' \
#          --dropout 0.2 \
#          --learning_rate 0.05 \
#          --scheduler_gamma 0.9 \
#          --batch_size 128 \
#          --gpus 1 \
#          --pwe 1

# ----------------------------------------------

# CATE_SOCIAL DATA (train)
#python models/CNN/trainer.py \
#          --train \
#          --data_dir "$data_dir" \
#          --word_vec_dir "$word_vec_dir" \
#          --dataset CATE_SOCIAL \
#          --embed_dim 200 \
#          --out_channels 256 \
#          --kernel_heights '2, 3, 4, 5' \
#          --dropout 0.2 \
#          --learning_rate 0.05 \
#          --scheduler_gamma 0.9 \
#          --batch_size 128 \
#          --gpus 1 \
#          --pwe 1

# CATE_ECONOMIC DATA (train)
#python models/CNN/trainer.py \
#          --train \
#          --data_dir "$data_dir" \
#          --word_vec_dir "$word_vec_dir" \
#          --dataset CATE_ECONOMIC \
#          --embed_dim 200 \
#          --out_channels 256 \
#          --kernel_heights '2, 3, 4, 5' \
#          --dropout 0.2 \
#          --learning_rate 0.05 \
#          --scheduler_gamma 0.9 \
#          --batch_size 128 \
#          --gpus 1 \
#          --pwe 1


# CATE_ENVIRONMENTAL DATA (train)
#python models/CNN/trainer.py \
#          --train \
#          --data_dir "$data_dir" \
#          --word_vec_dir "$word_vec_dir" \
#          --dataset CATE_ENVIRONMENTAL \
#          --embed_dim 200 \
#          --out_channels 256 \
#          --kernel_heights '2, 3, 4, 5' \
#          --dropout 0.2 \
#          --learning_rate 0.05 \
#          --scheduler_gamma 0.9 \
#          --batch_size 128 \
#          --gpus 1 \
#          --pwe 1


# CATE_OTHER DATA (train)
#python models/CNN/trainer.py \
#          --train \
#          --data_dir "$data_dir" \
#          --word_vec_dir "$word_vec_dir" \
#          --dataset CATE_OTHER \
#          --embed_dim 200 \
#          --out_channels 256 \
#          --kernel_heights '2, 3, 4, 5' \
#          --dropout 0.2 \
#          --learning_rate 0.05 \
#          --scheduler_gamma 0.9 \
#          --batch_size 128 \
#          --gpus 1 \
#          --pwe 1

# ----------------------------------------------

# STANCE_SOCIAL DATA (train)
#python models/CNN/trainer.py \
#          --train \
#          --data_dir "$data_dir" \
#          --word_vec_dir "$word_vec_dir" \
#          --dataset STANCE_SOCIAL \
#          --embed_dim 200 \
#          --out_channels 256 \
#          --kernel_heights '2, 3, 4, 5' \
#          --dropout 0.2 \
#          --learning_rate 0.05 \
#          --scheduler_gamma 0.9 \
#          --batch_size 128 \
#          --gpus 1 \
#          --pwe 1

# STANCE_ECONOMIC DATA (train)
#python models/CNN/trainer.py \
#          --train \
#          --data_dir "$data_dir" \
#          --word_vec_dir "$word_vec_dir" \
#          --dataset STANCE_ECONOMIC \
#          --embed_dim 200 \
#          --out_channels 256 \
#          --kernel_heights '2, 3, 4, 5' \
#          --dropout 0.2 \
#          --learning_rate 0.05 \
#          --scheduler_gamma 0.9 \
#          --batch_size 128 \
#          --gpus 1 \
#          --pwe 1


# STANCE_ENVIRONMENTAL DATA (train)
#python models/CNN/trainer.py \
#          --train \
#          --data_dir "$data_dir" \
#          --word_vec_dir "$word_vec_dir" \
#          --dataset STANCE_ENVIRONMENTAL \
#          --embed_dim 200 \
#          --out_channels 256 \
#          --kernel_heights '2, 3, 4, 5' \
#          --dropout 0.2 \
#          --learning_rate 0.05 \
#          --scheduler_gamma 0.9 \
#          --batch_size 128 \
#          --gpus 1 \
#          --pwe 1


# STANCE_OTHER DATA (train)
#python models/CNN/trainer.py \
#          --train \
#          --data_dir "$data_dir" \
#          --word_vec_dir "$word_vec_dir" \
#          --dataset STANCE_OTHER \
#          --embed_dim 200 \
#          --out_channels 256 \
#          --kernel_heights '2, 3, 4, 5' \
#          --dropout 0.2 \
#          --learning_rate 0.05 \
#          --scheduler_gamma 0.9 \
#          --batch_size 128 \
#          --gpus 1 \
#          --pwe 1

# ----------------------------------------------

# CATE_STANCE_SOCIAL DATA (train)
python models/CNN/trainer.py \
          --train \
          --data_dir "$data_dir" \
          --word_vec_dir "$word_vec_dir" \
          --dataset CATE_STANCE_SOCIAL \
          --embed_dim 200 \
          --out_channels 256 \
          --kernel_heights '2, 3, 4, 5' \
          --dropout 0.2 \
          --learning_rate 0.05 \
          --scheduler_gamma 0.9 \
          --batch_size 128 \
          --gpus 1 \
          --pwe 1

# CATE_STANCE_ECONOMIC DATA (train)
python models/CNN/trainer.py \
          --train \
          --data_dir "$data_dir" \
          --word_vec_dir "$word_vec_dir" \
          --dataset CATE_STANCE_ECONOMIC \
          --embed_dim 200 \
          --out_channels 256 \
          --kernel_heights '2, 3, 4, 5' \
          --dropout 0.2 \
          --learning_rate 0.05 \
          --scheduler_gamma 0.9 \
          --batch_size 128 \
          --gpus 1 \
          --pwe 1


# CATE_STANCE_ENVIRONMENTAL DATA (train)
python models/CNN/trainer.py \
          --train \
          --data_dir "$data_dir" \
          --word_vec_dir "$word_vec_dir" \
          --dataset CATE_STANCE_ENVIRONMENTAL \
          --embed_dim 200 \
          --out_channels 256 \
          --kernel_heights '2, 3, 4, 5' \
          --dropout 0.2 \
          --learning_rate 0.05 \
          --scheduler_gamma 0.9 \
          --batch_size 128 \
          --gpus 1 \
          --pwe 1


# CATE_STANCE_OTHER DATA (train)
python models/CNN/trainer.py \
          --train \
          --data_dir "$data_dir" \
          --word_vec_dir "$word_vec_dir" \
          --dataset CATE_STANCE_OTHER \
          --embed_dim 200 \
          --out_channels 256 \
          --kernel_heights '2, 3, 4, 5' \
          --dropout 0.2 \
          --learning_rate 0.05 \
          --scheduler_gamma 0.9 \
          --batch_size 128 \
          --gpus 1 \
          --pwe 1

# Grid Search
#python src/baselines/CNN/grid_search.py \
#          --dataset TREC \
#          --embed_dim 200 \
#          --gpus 1 \
#          --pwe 1