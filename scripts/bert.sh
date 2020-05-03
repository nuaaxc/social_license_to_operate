#!/usr/bin/env bash

source include.sh

# REL DATA
#python models/BERT/trainer.py \
#          --train \
#          --data_dir "$data_dir" \
#          --pretrained_model distilbert-base-uncased \
#          --dataset REL \
#          --learning_rate 2e-5 \
#          --batch_size 16 \
#          --max_length 128 \
#          --dropout 0.1 \
#          --gpus 1

#python models/BERT/trainer.py \
#          --test \
#          --error_analysis \
#          --data_dir "$data_dir" \
#          --pretrained_model distilbert-base-uncased \
#          --dataset REL \
#          --weight_path $src_path/logs_BERT_REL/version_0/checkpoints/epoch=0.ckpt \
#          --cfg_path $src_path/logs_BERT_REL/version_0/meta_tags.csv \
#          --learning_rate 1e-5 \
#          --batch_size 16 \
#          --max_length 128 \
#          --dropout 0.1 \
#          --gpus 1

# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------

# CATE_SOCIAL DATA
#python models/BERT/trainer.py \
#          --train \
#          --data_dir "$data_dir" \
#          --pretrained_model distilbert-base-uncased \
#          --dataset CATE_SOCIAL \
#          --learning_rate 1e-5 \
#          --batch_size 16 \
#          --max_length 128 \
#          --dropout 0.1 \
#          --gpus 1

#python models/BERT/trainer.py \
#          --test \
#          --error_analysis \
#          --data_dir "$data_dir" \
#          --pretrained_model distilbert-base-uncased \
#          --dataset CATE_SOCIAL \
#          --weight_path $src_path/logs_BERT_CATE_SOCIAL/version_0/checkpoints/epoch=2.ckpt \
#          --cfg_path $src_path/logs_BERT_CATE_SOCIAL/version_0/meta_tags.csv \
#          --learning_rate 1e-5 \
#          --batch_size 16 \
#          --max_length 128 \
#          --dropout 0.1 \
#          --gpus 1

# CATE_ECONOMIC DATA
#python models/BERT/trainer.py \
#          --train \
#          --data_dir "$data_dir" \
#          --pretrained_model distilbert-base-uncased \
#          --dataset CATE_ECONOMIC \
#          --learning_rate 1e-5 \
#          --batch_size 16 \
#          --max_length 128 \
#          --dropout 0.1 \
#          --gpus 1

#python models/BERT/trainer.py \
#          --test \
#          --error_analysis \
#          --data_dir "$data_dir" \
#          --pretrained_model distilbert-base-uncased \
#          --dataset CATE_ECONOMIC \
#          --weight_path $src_path/logs_BERT_CATE_ECONOMIC/version_0/checkpoints/epoch=7.ckpt \
#          --cfg_path $src_path/logs_BERT_CATE_ECONOMIC/version_0/meta_tags.csv \
#          --learning_rate 1e-5 \
#          --batch_size 16 \
#          --max_length 128 \
#          --dropout 0.1 \
#          --gpus 1

# CATE_ENVIRONMENTAL DATA
#python models/BERT/trainer.py \
#          --train \
#          --data_dir "$data_dir" \
#          --pretrained_model distilbert-base-uncased \
#          --dataset CATE_ENVIRONMENTAL \
#          --learning_rate 1e-5 \
#          --batch_size 16 \
#          --max_length 128 \
#          --dropout 0.1 \
#          --gpus 1

#python models/BERT/trainer.py \
#          --test \
#          --error_analysis \
#          --data_dir "$data_dir" \
#          --pretrained_model distilbert-base-uncased \
#          --dataset CATE_ENVIRONMENTAL \
#          --weight_path $src_path/logs_BERT_CATE_ENVIRONMENTAL/version_0/checkpoints/epoch=4.ckpt \
#          --cfg_path $src_path/logs_BERT_CATE_ENVIRONMENTAL/version_0/meta_tags.csv \
#          --learning_rate 1e-5 \
#          --batch_size 16 \
#          --max_length 128 \
#          --dropout 0.1 \
#          --gpus 1

# CATE_OTHER DATA
#python models/BERT/trainer.py \
#          --train \
#          --data_dir "$data_dir" \
#          --pretrained_model distilbert-base-uncased \
#          --dataset CATE_OTHER \
#          --learning_rate 1e-5 \
#          --batch_size 16 \
#          --max_length 128 \
#          --dropout 0.1 \
#          --gpus 1

#python models/BERT/trainer.py \
#          --test \
#          --error_analysis \
#          --data_dir "$data_dir" \
#          --pretrained_model distilbert-base-uncased \
#          --dataset CATE_OTHER \
#          --weight_path $src_path/logs_BERT_CATE_OTHER/version_0/checkpoints/epoch=4.ckpt \
#          --cfg_path $src_path/logs_BERT_CATE_OTHER/version_0/meta_tags.csv \
#          --learning_rate 1e-5 \
#          --batch_size 16 \
#          --max_length 128 \
#          --dropout 0.1 \
#          --gpus 1

# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------

# STANCE_SOCIAL DATA
#python models/BERT/trainer.py \
#          --train \
#          --data_dir "$data_dir" \
#          --pretrained_model distilbert-base-uncased \
#          --dataset STANCE_SOCIAL \
#          --learning_rate 1e-5 \
#          --batch_size 16 \
#          --max_length 128 \
#          --dropout 0.1 \
#          --gpus 1

#python models/BERT/trainer.py \
#          --test \
#          --error_analysis \
#          --data_dir "$data_dir" \
#          --pretrained_model distilbert-base-uncased \
#          --dataset STANCE_SOCIAL \
#          --weight_path $src_path/logs_BERT_STANCE_SOCIAL/version_0/checkpoints/epoch=1.ckpt \
#          --cfg_path $src_path/logs_BERT_STANCE_SOCIAL/version_0/meta_tags.csv \
#          --learning_rate 1e-5 \
#          --batch_size 16 \
#          --max_length 128 \
#          --dropout 0.1 \
#          --gpus 1

# STANCE_ECONOMIC DATA
#python models/BERT/trainer.py \
#          --train \
#          --data_dir "$data_dir" \
#          --pretrained_model distilbert-base-uncased \
#          --dataset STANCE_ECONOMIC \
#          --learning_rate 1e-5 \
#          --batch_size 16 \
#          --max_length 128 \
#          --dropout 0.1 \
#          --gpus 1

#python models/BERT/trainer.py \
#          --test \
#          --error_analysis \
#          --data_dir "$data_dir" \
#          --pretrained_model distilbert-base-uncased \
#          --dataset STANCE_ECONOMIC \
#          --weight_path $src_path/logs_BERT_STANCE_ECONOMIC/version_0/checkpoints/epoch=1.ckpt \
#          --cfg_path $src_path/logs_BERT_STANCE_ECONOMIC/version_0/meta_tags.csv \
#          --learning_rate 1e-5 \
#          --batch_size 16 \
#          --max_length 128 \
#          --dropout 0.1 \
#          --gpus 1

# STANCE_ENVIRONMENTAL DATA
#python models/BERT/trainer.py \
#          --train \
#          --data_dir "$data_dir" \
#          --pretrained_model distilbert-base-uncased \
#          --dataset STANCE_ENVIRONMENTAL \
#          --learning_rate 1e-5 \
#          --batch_size 16 \
#          --max_length 128 \
#          --dropout 0.1 \
#          --gpus 1

#python models/BERT/trainer.py \
#          --test \
#          --error_analysis \
#          --data_dir "$data_dir" \
#          --pretrained_model distilbert-base-uncased \
#          --dataset STANCE_ENVIRONMENTAL \
#          --weight_path $src_path/logs_BERT_STANCE_ENVIRONMENTAL/version_0/checkpoints/epoch=1.ckpt \
#          --cfg_path $src_path/logs_BERT_STANCE_ENVIRONMENTAL/version_0/meta_tags.csv \
#          --learning_rate 1e-5 \
#          --batch_size 16 \
#          --max_length 128 \
#          --dropout 0.1 \
#          --gpus 1


# STANCE_OTHER DATA
#python models/BERT/trainer.py \
#          --train \
#          --data_dir "$data_dir" \
#          --pretrained_model distilbert-base-uncased \
#          --dataset STANCE_OTHER \
#          --learning_rate 1e-5 \
#          --batch_size 16 \
#          --max_length 128 \
#          --dropout 0.1 \
#          --gpus 1

#python models/BERT/trainer.py \
#          --test \
#          --error_analysis \
#          --data_dir "$data_dir" \
#          --pretrained_model distilbert-base-uncased \
#          --dataset STANCE_OTHER \
#          --weight_path $src_path/logs_BERT_STANCE_OTHER/version_0/checkpoints/epoch=1.ckpt \
#          --cfg_path $src_path/logs_BERT_STANCE_OTHER/version_0/meta_tags.csv \
#          --learning_rate 1e-5 \
#          --batch_size 16 \
#          --max_length 128 \
#          --dropout 0.1 \
#          --gpus 1

# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------

# CATE_STANCE_SOCIAL DATA
#python models/BERT/trainer.py \
#          --train \
#          --data_dir "$data_dir" \
#          --pretrained_model distilbert-base-uncased \
#          --dataset CATE_STANCE_SOCIAL \
#          --learning_rate 1e-5 \
#          --batch_size 16 \
#          --max_length 128 \
#          --dropout 0.1 \
#          --gpus 1

#python models/BERT/trainer.py \
#          --test \
#          --error_analysis \
#          --data_dir "$data_dir" \
#          --pretrained_model distilbert-base-uncased \
#          --dataset CATE_STANCE_SOCIAL \
#          --weight_path $src_path/logs_BERT_CATE_STANCE_SOCIAL/version_0/checkpoints/epoch=1.ckpt \
#          --cfg_path $src_path/logs_BERT_CATE_STANCE_SOCIAL/version_0/meta_tags.csv \
#          --learning_rate 1e-5 \
#          --batch_size 16 \
#          --max_length 128 \
#          --dropout 0.1 \
#          --gpus 1

# CATE_STANCE_ECONOMIC DATA
#python models/BERT/trainer.py \
#          --train \
#          --data_dir "$data_dir" \
#          --pretrained_model distilbert-base-uncased \
#          --dataset CATE_STANCE_ECONOMIC \
#          --learning_rate 1e-5 \
#          --batch_size 16 \
#          --max_length 128 \
#          --dropout 0.1 \
#          --gpus 1

#python models/BERT/trainer.py \
#          --test \
#          --error_analysis \
#          --data_dir "$data_dir" \
#          --pretrained_model distilbert-base-uncased \
#          --dataset CATE_STANCE_ECONOMIC \
#          --weight_path $src_path/logs_BERT_CATE_STANCE_ECONOMIC/version_0/checkpoints/epoch=1.ckpt \
#          --cfg_path $src_path/logs_BERT_CATE_STANCE_ECONOMIC/version_0/meta_tags.csv \
#          --learning_rate 1e-5 \
#          --batch_size 16 \
#          --max_length 128 \
#          --dropout 0.1 \
#          --gpus 1

# CATE_STANCE_ENVIRONMENTAL DATA
#python models/BERT/trainer.py \
#          --train \
#          --data_dir "$data_dir" \
#          --pretrained_model distilbert-base-uncased \
#          --dataset CATE_STANCE_ENVIRONMENTAL \
#          --learning_rate 1e-5 \
#          --batch_size 16 \
#          --max_length 128 \
#          --dropout 0.1 \
#          --gpus 1

#python models/BERT/trainer.py \
#          --test \
#          --error_analysis \
#          --data_dir "$data_dir" \
#          --pretrained_model distilbert-base-uncased \
#          --dataset CATE_STANCE_ENVIRONMENTAL \
#          --weight_path $src_path/logs_BERT_CATE_STANCE_ENVIRONMENTAL/version_0/checkpoints/epoch=1.ckpt \
#          --cfg_path $src_path/logs_BERT_CATE_STANCE_ENVIRONMENTAL/version_0/meta_tags.csv \
#          --learning_rate 1e-5 \
#          --batch_size 16 \
#          --max_length 128 \
#          --dropout 0.1 \
#          --gpus 1

# CATE_STANCE_OTHER DATA
#python models/BERT/trainer.py \
#          --train \
#          --data_dir "$data_dir" \
#          --pretrained_model distilbert-base-uncased \
#          --dataset CATE_STANCE_OTHER \
#          --learning_rate 1e-5 \
#          --batch_size 16 \
#          --max_length 128 \
#          --dropout 0.1 \
#          --gpus 1

#python models/BERT/trainer.py \
#          --test \
#          --error_analysis \
#          --data_dir "$data_dir" \
#          --pretrained_model distilbert-base-uncased \
#          --dataset CATE_STANCE_OTHER \
#          --weight_path $src_path/logs_BERT_CATE_STANCE_OTHER/version_0/checkpoints/epoch=1.ckpt \
#          --cfg_path $src_path/logs_BERT_CATE_STANCE_OTHER/version_0/meta_tags.csv \
#          --learning_rate 1e-5 \
#          --batch_size 16 \
#          --max_length 128 \
#          --dropout 0.1 \
#          --gpus 1
