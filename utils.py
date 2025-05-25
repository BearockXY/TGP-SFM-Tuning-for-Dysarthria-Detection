import numpy as np
import os
import random
import glob
from dataset_dir import *
import torch
import torch.nn.functional as F
import torch.nn as nn
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def generate_speaker_sampling_preset():
    fold_info = np.load('10fold_speaker_info.npy',allow_pickle=True)
    return fold_info

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        # euclidean_distance = F.cosine_similarity(output1, output2)
        # euclidean_distance = torch.sum(torch.abs(output1 - output2))
        loss_contrastive = torch.mean((label) * torch.pow(euclidean_distance, 2) + \
                                      (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        # modified version
        # loss_contrastive = torch.mean( 0.2 * (label) * torch.pow(euclidean_distance, 2) + (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        # loss_contrastive = torch.mean( (label) * torch.pow(euclidean_distance, 2) )
                                    #   + 0.5 * (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        # loss_contrastive = torch.mean( (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive

def make_tSNE(out_feature,out_label,figure_title,average_loss,random_state):
    out_feature_np = out_feature[0]
    for i in range(len(out_feature) - 1 ):
        out_feature_np = np.concatenate((out_feature_np, out_feature[i+1]))

    out_label_np = out_label[0]
    for i in range(len(out_label) - 1 ):
        if len(out_label[i+1].shape) == 0:
            continue
        out_label_np = np.concatenate((out_label_np, out_label[i+1]))
        # print(out_label_np)
        # print(out_label[i+1])
    # print(out_label_np)
        
    tsne = TSNE(n_components=2, random_state=random_state)
    activations_2d = tsne.fit_transform(out_feature_np)
    # print(activations_2d)
    plt.figure(figsize=(10, 6))
    for class_idx in np.unique(out_label_np):
        indices = np.where(out_label_np == class_idx)
        plt.scatter(activations_2d[indices, 0], activations_2d[indices, 1], label=f'Class {class_idx}')
    plt.legend()
    title = 'loss = ' + str(average_loss)
    plt.title(title)
    plt.savefig(figure_title)
    plt.close()

def make_tSNE_filtered(out_feature,out_label,figure_title,average_loss,random_state,pair):

    out_feature_np = out_feature[0]
    for i in range(len(out_feature) - 1 ):
        out_feature_np = np.concatenate((out_feature_np, out_feature[i+1]))

    out_label_np = out_label[0]
    for i in range(len(out_label) - 1 ):
        if len(out_label[i+1].shape) == 0:
            continue
        out_label_np = np.concatenate((out_label_np, out_label[i+1]))
        # print(out_label_np)
        # print(out_label[i+1])


    tsne = TSNE(n_components=2, random_state=random_state)
    activations_2d = tsne.fit_transform(out_feature_np)

    # print(activations_2d)
    plt.figure(figsize=(10, 6))
    for class_idx in np.unique(out_label_np):
        if class_idx not in pair:continue
        indices = np.where(out_label_np == class_idx)
        plt.scatter(activations_2d[indices, 0], activations_2d[indices, 1], label=f'Class {class_idx}')
    plt.legend()
    title = 'loss = ' + str(average_loss)
    plt.title(title)
    plt.savefig(figure_title)
    plt.close()

def make_tSNE_filtered_V2(out_feature,out_label,figure_title,average_loss,random_state,pair):

    out_feature_np = out_feature[0]
    for i in range(len(out_feature) - 1 ):
        out_feature_np = np.concatenate((out_feature_np, out_feature[i+1]))

    out_label_np = out_label[0]
    for i in range(len(out_label) - 1 ):
        if len(out_label[i+1].shape) == 0:
            continue
        out_label_np = np.concatenate((out_label_np, out_label[i+1]))
        # print(out_label_np)
        # print(out_label[i+1])
    # print(out_label_np)

    mask = np.isin(out_label_np, pair)
    filtered_features = out_feature_np[mask]
    filtered_labels = out_label_np[mask]

    tsne = TSNE(n_components=2, random_state=random_state)
    activations_2d = tsne.fit_transform(filtered_features)
    # print(activations_2d)
    plt.figure(figsize=(10, 6))
    for class_idx in np.unique(filtered_labels):
        indices = np.where(filtered_labels == class_idx)
        plt.scatter(activations_2d[indices, 0], activations_2d[indices, 1], label=f'Class {class_idx}')
    plt.legend()
    title = 'loss = ' + str(average_loss)
    plt.title(title)
    plt.savefig(figure_title)
    plt.close()


# Training sample list loading========================================================================================
def load_ALS_from_folder():
    als_root = "/media/bearock/ssd_Speech/data/R01_dataset/ALS_local_processed"
    wav_files = glob.glob(os.path.join(als_root, '**', '*.wav'), recursive=True)
    als_sample_array = []
    for wav_dir in wav_files:
        info = wav_dir.split('/')
        speakerID = info[7]
        desease_type = 'ALS'
        gender = speakerID[3]
        type_of_sample = 'clear_sentences'

        if 'rhythm1' in wav_dir or 'R_1' in wav_dir:
            script = 'The supermarket chain shut down because of poor management'
        elif 'rhythm2' in wav_dir or 'R_2' in wav_dir:
            script = 'Much more money must be donated to make this department succeed'
        elif 'rhythm3' in wav_dir or 'R_3' in wav_dir:
            script = 'In this famous coffee shop they serve the best doughnuts in town'
        elif 'rhythm4' in wav_dir or 'R_4' in wav_dir:
            script = 'The chairman decided to pave over the shopping center garden'
        elif 'rhythm5' in wav_dir or 'R_5' in wav_dir:
            script = 'The standards committee met this afternoon in an open meeting'
        else:
            print('not found')

        als_sample_array.append([wav_dir,speakerID,desease_type,gender,type_of_sample,script])
    return als_sample_array


def load_R01_from_folder():
    R01_root = get_R01_root()
    Dysarthric_dir = R01_root+'/Dysarthric_Data_Final'
    Healthy_dir = R01_root + '/Healthy_Data_Final'
    
    Dysarthric_ID_list = os.listdir(Dysarthric_dir)
    Dysarthric_ID_list.sort()
    Dysarthric_sample_list = create_R01_sample_list(Dysarthric_ID_list,Dysarthric_dir)

    Healthy_ID_list = os.listdir(Healthy_dir)
    Healthy_ID_list.sort()
    Healthy_sample_list = create_R01_sample_list(Healthy_ID_list,Healthy_dir)
    return Dysarthric_sample_list, Healthy_sample_list

def create_R01_sample_list(ID_list, root_dir):
    sample_list = []
    for user_folder in ID_list:

        user_ID,disease_type, gender, data_type = get_index_R01(user_folder)

        folder_dir = root_dir + '/' + user_folder
        file_extension = '.wav'  # Example for text files
        wavefile_list = [f for f in os.listdir(folder_dir) if f.endswith(file_extension)]
        for wavefile in wavefile_list:
            wave_dir = folder_dir + '/' + wavefile
            script_name = wavefile[0:-4] + '.lab'
            script_dir = folder_dir + '/' + script_name
            script = np.loadtxt(script_dir, delimiter=',',dtype = str)
            sample = [wave_dir,user_ID,disease_type, gender, data_type,script.item()]
            sample_list.append(sample)
    return sample_list

def get_index_R01(filename):
    dash_index_array = []
    for i in range(len(filename)):
        if(filename[i]=='_'):
            dash_index_array.append(i)
    user_ID = filename[0:dash_index_array[0]]
    
    number_index = 0
    for i in range(len(user_ID)):
        if user_ID[i].isnumeric():
            number_index = i
            break

    data_type = filename[dash_index_array[1]+1:]
    disease_type = user_ID[0:number_index-1]
    gender = user_ID[number_index-1:number_index]
    return user_ID,disease_type, gender, data_type

def create_R01_sentence_only_sample_list(ID_list, root_dir):
    sample_list = []
    for user_folder in ID_list:
        if user_folder[-9:] != 'sentences':
            continue

        user_ID,disease_type, gender, data_type = get_index_R01(user_folder)

        folder_dir = root_dir + '/' + user_folder
        file_extension = '.wav'  # Example for text files
        wavefile_list = [f for f in os.listdir(folder_dir) if f.endswith(file_extension)]
        for wavefile in wavefile_list:
            wave_dir = folder_dir + '/' + wavefile
            script_name = wavefile[0:-4] + '.lab'
            script_dir = folder_dir + '/' + script_name
            script = np.loadtxt(script_dir, delimiter=',',dtype = str)
            sample = [wave_dir,user_ID,disease_type, gender, data_type,script.item()]
            sample_list.append(sample)
    return sample_list

def load_R01_sentence_only_from_folder():
    R01_root = get_R01_root()
    Dysarthric_dir = R01_root+'/Dysarthric_Data_Final'
    Healthy_dir = R01_root + '/Healthy_Data_Final'
    
    Dysarthric_ID_list = os.listdir(Dysarthric_dir)
    Dysarthric_ID_list.sort()
    Dysarthric_sample_list = create_R01_sentence_only_sample_list(Dysarthric_ID_list,Dysarthric_dir)

    Healthy_ID_list = os.listdir(Healthy_dir)
    Healthy_ID_list.sort()
    Healthy_sample_list = create_R01_sentence_only_sample_list(Healthy_ID_list,Healthy_dir)
    return Dysarthric_sample_list, Healthy_sample_list

# ================================================================================================================================

def split_list_by_participant_preset(input_list, user_list_train,user_list_eval,user_list_test):
  
    split_list_train = []
    split_list_eval = []
    split_list_test = []
    # print('================================= speaker info ================================')
    # print('training speakers', user_list_train)
    # print('evaluation speakers', user_list_eval)
    # print('testing speakers', user_list_test)

    for i in range(len(input_list)):
        Speaker_ID = input_list[i][1]
        if Speaker_ID in user_list_train:
            split_list_train.append(input_list[i])
        elif Speaker_ID in user_list_eval:
            split_list_eval.append(input_list[i])
        elif Speaker_ID in user_list_test:
            split_list_test.append(input_list[i])
        # else:
        #     raise ValueError("unknown speaker ID.")

    # Split the list
    return split_list_train, split_list_eval, split_list_test
