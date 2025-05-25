#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--epochs', type=int, default=100, help="rounds of training")
    parser.add_argument('--bs', type=int, default=32, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.0001, help="learning rate")
    parser.add_argument('--weight_decay', type=float, default=1e-5, help="weight_decay")

    parser.add_argument('--wave_length', type=int, default=48000, help="waveform file length")
    parser.add_argument('--sample_per_record', type=int, default=10, help="sample_per_record")

    parser.add_argument("--save_dir", type=str, default='./test_result/Multi_TSNE_PairCheck_sentenceOnly_seperateTSN_2', help='configuration file path')
    parser.add_argument('--continue_training', action='store_true',default=False,  help='if True, local updating evenly distributed.')

   # parser.add_argument('--local_ep', type=int, default=5, help="the number of local epochs: E")
    # parser.add_argument('--local_bs', type=int, default=10, help="local batch size: B")
    # parser.add_argument('--sample_length', type=int, default=5, help="learning rate")
    # parser.add_argument('--server_lr', type=float, default=0.01, help="learning rate")
    # parser.add_argument('--momentum', type=float, default=0.9, help="SGD momentum (default: 0.9)")
    # parser.add_argument('--sampler',  type=str, default='uniform', choices=['uniform','normal','iid'], help='model name')
    # parser.add_argument('--save_dir',  type=str, default='save')

    # # model arguments
    # parser.add_argument('--model', type=str, default='resnet', help='model name')
    # parser.add_argument('--norm', type=str, default='batch_norm', help="batch_norm, group_norm, or None")

    # # other arguments
    # parser.add_argument('--dataset', type=str, default='cifar', help="name of dataset")
    # parser.add_argument('--iid', action='store_true', help='whether i.i.d or not')
    # parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    # parser.add_argument('--verbose', action='store_true', help='verbose print')
    # parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')

    # # sfl arguments
    # parser.add_argument('--cut', type=int, default=1, help="cut layer")
    # parser.add_argument('--buffer_size_in_mult', type=int, default=1, help="size of server-side local buffer in mulitple of raw data size")

    # # sfl techniques
    # # client_freeze
    # parser.add_argument('--freeze_start_epoch', type=int, default=200, help="which epoch stars freezing model")
    # parser.add_argument('--global_update', type=int, default=0, help=" how many epochs to do the global updating  ")
    # parser.add_argument('--local_update', type=int, default=0, help=" how many epochs to do the local updating  ")


    # parser.add_argument('--even', action='store_true',default=False,  help='if True, local updating evenly distributed.')
    # parser.add_argument('--acsending', action='store_true',default=False,  help='if True, local updating distributed more later.')
    # parser.add_argument('--descending', action='store_true',default=False,  help='if True, local updating distributed more at the begining.')

    # # MMER args
    # parser.add_argument("--config", type=str, required=True, help='configuration file path')
    # parser.add_argument("--bert_config", type=str, required=True, help='configuration file path for BERT')
    # parser.add_argument("--epochs", type=int, default=100, help="training epoches")
    # parser.add_argument("--csv_path", type=str, required=True, help="path of csv")
    # parser.add_argument("--save_path", type=str, default="./", help="report or ckpt save path")
    # parser.add_argument("--data_path_audio", type=str, required=True, help="path to raw audio wav files")
    # parser.add_argument("--data_path_roberta", type=str, required=True, help="path to roberta embeddings for text")
    # parser.add_argument("--data_path_audio_augmented", type=str, required=True, help="path to augmented audio wav files")
    # parser.add_argument("--data_path_roberta_augmented", type=str, required=True, help="path to augmented roberta embeddings for text")
    # parser.add_argument("--learning_rate", type=float, default=5e-5, help="learning rate for the specific run")
    # parser.add_argument("--batch_size", type=int, default=2, help="batch size")
    # parser.add_argument("--accum_grad", type=int, default=4, help="gradient accumulation steps")
    # parser.add_argument("--lambda_value", type=float, default=0.1, help="lambda_value to weight the auxiliary losses")
    # parser.add_argument("--num_workers", type=int, default=4, help="number of workers in data loader")

    args = parser.parse_args()
    print(args)
    return args
