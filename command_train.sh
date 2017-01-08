#!/usr/bin/env bash
python3 train.py --hidden 256 --num_classes 11 --batch_size 128 --seq_length 49 --seq_dim 13 --num_layers 2 --epoch 100 --lr 0.003 --decay 0.97 --data_set_load static/trn_data_normal.pkl static/trn_label_normal.pkl static/trn_cluster_normal.pkl
mv max_model.ckpt max_model_trn_normal.ckpt
python3 train.py --hidden 256 --num_classes 11 --batch_size 128 --seq_length 49 --seq_dim 13 --num_layers 2 --epoch 100 --lr 0.003 --decay 0.97 --data_set_load static/trl_data_normal.pkl static/trl_label_normal.pkl static/trl_cluster_normal.pkl
mv max_model.ckpt max_model_trl_normal.ckpt
python3 train.py --hidden 256 --num_classes 11 --batch_size 128 --seq_length 49 --seq_dim 13 --num_layers 2 --epoch 100 --lr 0.003 --decay 0.97 --data_set_load static/trr_data_normal.pkl static/trr_label_normal.pkl static/trr_cluster_normal.pkl
mv max_model.ckpt max_model_trr_normal.ckpt
python3 train.py --hidden 256 --num_classes 11 --batch_size 128 --seq_length 49 --seq_dim 13 --num_layers 2 --epoch 100 --lr 0.003 --decay 0.97 --data_set_load static/trs_data_normal.pkl static/trs_label_normal.pkl static/trs_cluster_normal.pkl
mv max_model.ckpt max_model_trs_normal.ckpt

python3 train.py --hidden 256 --num_classes 11 --batch_size 128 --seq_length 49 --seq_dim 13 --num_layers 2 --epoch 100 --lr 0.003 --decay 0.97 --data_set_load static/trn_data_bass.pkl static/trn_label_bass.pkl static/trn_cluster_bass.pkl
mv max_model.ckpt max_model_trn_bass.ckpt
python3 train.py --hidden 256 --num_classes 11 --batch_size 128 --seq_length 49 --seq_dim 13 --num_layers 2 --epoch 100 --lr 0.003 --decay 0.97 --data_set_load static/trn_data_drums.pkl static/trn_label_drums.pkl static/trn_cluster_drums.pkl
mv max_model.ckpt max_model_trn_drums.ckpt
python3 train.py --hidden 256 --num_classes 11 --batch_size 128 --seq_length 49 --seq_dim 13 --num_layers 2 --epoch 100 --lr 0.003 --decay 0.97 --data_set_load static/trn_data_melody.pkl static/trn_label_melody.pkl static/trn_cluster_melody.pkl
mv max_model.ckpt max_model_trn_melody.ckpt
python3 train.py --hidden 256 --num_classes 11 --batch_size 128 --seq_length 49 --seq_dim 13 --num_layers 2 --epoch 100 --lr 0.003 --decay 0.97 --data_set_load static/trn_data_other.pkl static/trn_label_other.pkl static/trn_cluster_other.pkl
mv max_model.ckpt max_model_trn_other.ckpt