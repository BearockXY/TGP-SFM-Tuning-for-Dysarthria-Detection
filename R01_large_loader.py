from Model_fundation import *
from utils import *
from data_preparing import prepare_sample
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from data_preparing import create_training_set_from_list
import os

def get_wav_files(root_folder):
    wav_files = []

    # Traverse through root folder
    for speaker_ID in os.listdir(root_folder):
        speaker_folder = os.path.join(root_folder, speaker_ID)
        if os.path.isdir(speaker_folder):
            # Iterate through files in speaker's folder
            for file_name in os.listdir(speaker_folder):
                if file_name.endswith(".wav") or file_name.endswith(".WAV"):
                    file_path = os.path.join(speaker_folder, file_name)
                    wav_files.append([speaker_ID, file_path])
                    # wav_files.append(file_path)

    return wav_files

def get_index_R01(filename):
    splited_name = filename.split('/')
    user_ID = splited_name[-2]
    
    number_index = 0
    for i in range(len(user_ID)):
        if user_ID[i].isnumeric():
            number_index = i
            break

    # data_type = filename[dash_index_array[1]+1:]
    disease_type = user_ID[0:number_index-1]
    gender = user_ID[number_index-1:number_index]
    return disease_type, gender

def create_R01_large_sample_list(wav_files_list):
    sample_list = []

    for speaker_ID,wavefile in wav_files_list:
        wave_dir = wavefile
        disease_type, gender = get_index_R01(wave_dir)
        sample = [wave_dir,speaker_ID,disease_type, gender,'none','none']
        sample_list.append(sample)
    return sample_list


def load_R01_large_from_folder():
    root_folder_DY = '/media/bearock/ssd_Speech/data/R01_large/dysarthria'
    wav_files_list_DY = get_wav_files(root_folder_DY)
    Dysarthric_sample_list = create_R01_large_sample_list(wav_files_list_DY)

    root_folder_HC = '/media/bearock/ssd_Speech/data/R01_large/healthy'
    wav_files_list_HC = get_wav_files(root_folder_HC)
    Healthy_sample_list = create_R01_large_sample_list(wav_files_list_HC)
    return Dysarthric_sample_list, Healthy_sample_list


def load_wav_from_dir_list(file_list):
    list_with_wav = []
    for sample in file_list:
        wav = prepare_sample(sample[0])
        sample_wav = [
            wav,
            sample[1],
            sample[2],
            sample[3],
            sample[4],
            sample[5]
            ]
        list_with_wav.append(sample_wav)
    return list_with_wav

def get_speaker_sampling_from_list(sample_list_all):
    speakers = []
    for sample in sample_list_all:
        speaker_ID = sample[1]
        if speaker_ID not in speakers:
            speakers.append(speaker_ID)
    
    random.shuffle(speakers)
    split_1 = int(len(speakers)*8/10)
    split_2 = split_1 + int(len(speakers)/10)
    speakers_train = speakers[0:split_1]
    speakers_eval = speakers[split_1:split_2]
    speakers_test = speakers[split_2:]
    
    return speakers_train, speakers_eval, speakers_test


def get_dataset_large_version1():
    Dysarthric_sample_list_all, Healthy_sample_list_all = load_R01_large_from_folder()
    user_sampling = generate_speaker_sampling_preset()

    # assigning speakers to different sets
    speaker_HC_list_train_for_DytDetec, speaker_HC_list_eval_for_DytDetec, speaker_HC_list_test_for_DytDetec = get_speaker_sampling_from_list(Healthy_sample_list_all)
    Speaker_Dysarthric_list_train_all,Speaker_Dysarthric_list_eval_all, Speaker_Dysarthric_list_test_all = get_speaker_sampling_from_list(Dysarthric_sample_list_all)
    
    # sorting the file lists via speakers

    Healthy_sample_DytDetec_list_train_all, \
        Healthy_sample_DytDetec_list_eval_all, \
            Healthy_sample_DytDetec_list_test_all = split_list_by_participant_preset(Healthy_sample_list_all, speaker_HC_list_train_for_DytDetec, speaker_HC_list_eval_for_DytDetec, speaker_HC_list_test_for_DytDetec)
    Healthy_sample_DytDetec_list_train_all = load_wav_from_dir_list(Healthy_sample_DytDetec_list_train_all)
    Healthy_sample_DytDetec_list_eval_all = load_wav_from_dir_list(Healthy_sample_DytDetec_list_eval_all)
    Healthy_sample_DytDetec_list_test_all = load_wav_from_dir_list(Healthy_sample_DytDetec_list_test_all)

    Dyt_sample_list_train_all, \
        Dyt_sample_list_eval_all, \
            Dyt_sample_list_test_all = split_list_by_participant_preset(Dysarthric_sample_list_all, Speaker_Dysarthric_list_train_all,Speaker_Dysarthric_list_eval_all, Speaker_Dysarthric_list_test_all)
    Dyt_sample_list_train_all = load_wav_from_dir_list(Dyt_sample_list_train_all)
    Dyt_sample_list_eval_all = load_wav_from_dir_list(Dyt_sample_list_eval_all)
    Dyt_sample_list_test_all = load_wav_from_dir_list(Dyt_sample_list_test_all)

    # ALS_Detection_training_file_list = [ALS_sample_list_train_all, Healthy_sample_ALSDetec_list_train_all]
    # ALS_Detection_evaluation_file_list = [ALS_sample_list_eval_all, Healthy_sample_ALSDetec_list_eval_all]

    # Dyt_Detection_training_file_list = [Dyt_sample_list_train_all, Healthy_sample_DytDetec_list_train_all]
    # Dyt_Detection_evaluation_file_list = [Dyt_sample_list_eval_all, Healthy_sample_DytDetec_list_eval_all]
    Dyt_Detection_testing_file_list = [Dyt_sample_list_test_all, Healthy_sample_DytDetec_list_test_all]

    Training_file_list = [Dyt_sample_list_train_all,Healthy_sample_DytDetec_list_train_all]
    Evaluation_file_list = [Dyt_sample_list_eval_all, Healthy_sample_DytDetec_list_eval_all]

    Training_wav_list = create_training_set_from_list(Training_file_list)
    Evaluation_wav_list = create_training_set_from_list(Evaluation_file_list)
    Testing_wav_list = create_training_set_from_list(Dyt_Detection_testing_file_list)

    return Training_wav_list, Evaluation_wav_list,Testing_wav_list