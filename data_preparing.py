from dataset_dir import *
from Model_fundation import *
from utils import *

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


class CustomDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data, label, SpeakerID, D_type, gender = self.samples[idx]
        # Ensure data is a tensor with shape (1, x)
        data = torch.tensor(data, dtype=torch.float)
        label = torch.tensor(label, dtype=torch.long)
        SpeakerID = torch.tensor(SpeakerID, dtype=torch.long)
        D_type = torch.tensor(D_type, dtype=torch.long)
        gender = torch.tensor(gender, dtype=torch.long)
        return data, label, SpeakerID, D_type, gender

def collate_fn(batch, target_length):
    # Separate data and labels
    data, labels, SpeakerID, D_type, gender = zip(*batch)

    # Pad or crop each sequence to the target length
    padded_data = []
    for seq in data:
        # print(seq.shape)
        seq_length = seq.size(1)  # Get the second dimension length
        if seq_length < target_length:
            # Pad the sequence with zeros if it's shorter than the target length
            padding = torch.zeros(1, target_length - seq_length)
            padded_seq = torch.cat([seq, padding], dim=1)
        else:
            # Crop the sequence if it's longer than the target length
            padded_seq = seq[:, :target_length]
        padded_data.append(padded_seq)

    # Stack the padded sequences and labels into tensors
    padded_data = torch.stack(padded_data)
    labels = torch.stack(labels)
    SpeakerID = torch.stack(SpeakerID)
    D_type = torch.stack(D_type)
    gender = torch.stack(gender)

    return padded_data, labels, SpeakerID, D_type, gender

def prepare_sample(audio_path, target_sample_rate=16000):
    # Load audio file
    waveform, sample_rate = torchaudio.load(audio_path)

    # Resample if necessary
    if sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
        waveform = resampler(waveform)

    # Normalize waveform
    waveform = waveform.mean(dim=0).unsqueeze(0)  # Make it mono and add batch dimension
    waveform = waveform / waveform.abs().max()  # Normalize to [-1, 1]

    return waveform

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

# def create_training_set_from_list(sample_list):
#     training_list = []
#     for i in range(len(sample_list)):
#         sample_list_onetype = sample_list[i]
#         if i == 0:
#             label = 1
#         else:
#             label = 0
#         for sample in sample_list_onetype:
#             waveform = sample[0]
#             SpeakerID = sample[1]
#             D_type = sample[2]
#             gender = sample[3]
#             training_list.append([waveform, label, SpeakerID, D_type, gender])
#     return training_list

def create_training_set_from_list(sample_list):
    training_list = []

    # Initialize dictionaries for index encoding
    speakerID_encoding = {}
    dtype_encoding = {}
    gender_encoding = {}
    
    # Counters to assign new indices
    speakerID_counter = 0
    dtype_counter = 0
    gender_counter = 0

    for i in range(len(sample_list)):
        sample_list_onetype = sample_list[i]
        
        # Assign label based on type (first set as label 1, rest as label 0)
        if i == 0:
            label = 1
        else:
            label = 0

        for sample in sample_list_onetype:
            waveform = sample[0]
            SpeakerID = sample[1]
            D_type = sample[2]
            gender = sample[3]

            # Encode SpeakerID
            if SpeakerID not in speakerID_encoding:
                speakerID_encoding[SpeakerID] = speakerID_counter
                speakerID_counter += 1
            encoded_SpeakerID = speakerID_encoding[SpeakerID]

            # Encode D_type
            if D_type not in dtype_encoding:
                dtype_encoding[D_type] = dtype_counter
                dtype_counter += 1
            encoded_D_type = dtype_encoding[D_type]

            # Encode gender
            if gender not in gender_encoding:
                gender_encoding[gender] = gender_counter
                gender_counter += 1
            encoded_gender = gender_encoding[gender]

            # Append the encoded sample
            training_list.append([waveform, label, encoded_SpeakerID, encoded_D_type, encoded_gender])

    # Print the encoding rules
    print("SpeakerID encoding:", speakerID_encoding)
    print("D_type encoding:", dtype_encoding)
    print("Gender encoding:", gender_encoding)

    return training_list

def create_training_set_from_list_sentence(sample_list, sample_length, num_samples_per_waveform=5):
    training_list = []

    # Initialize dictionaries for index encoding
    speakerID_encoding = {}
    dtype_encoding = {}
    gender_encoding = {}
    
    # Counters to assign new indices
    speakerID_counter = 0
    dtype_counter = 0
    gender_counter = 0

    for i in range(len(sample_list)):
        sample_list_onetype = sample_list[i]
        
        # Assign label based on type (first set as label 1, rest as label 0)
        label = 1 if i == 0 else 0

        for sample in sample_list_onetype:
            waveform = sample[0]  # Assuming waveform is a torch tensor with shape [1, x]
            SpeakerID = sample[1]
            D_type = sample[2]
            gender = sample[3]

            # Encode SpeakerID
            if SpeakerID not in speakerID_encoding:
                speakerID_encoding[SpeakerID] = speakerID_counter
                speakerID_counter += 1
            encoded_SpeakerID = speakerID_encoding[SpeakerID]

            # Encode D_type
            if D_type not in dtype_encoding:
                dtype_encoding[D_type] = dtype_counter
                dtype_counter += 1
            encoded_D_type = dtype_encoding[D_type]

            # Encode gender
            if gender not in gender_encoding:
                gender_encoding[gender] = gender_counter
                gender_counter += 1
            encoded_gender = gender_encoding[gender]

            # Sample random segments from the waveform
            waveform_length = waveform.shape[1]  # Assuming shape is [1, x]
            for _ in range(num_samples_per_waveform):
                if waveform_length < sample_length:
                    # If waveform is too short, pad with zeros at the end to reach sample_length
                    padding_length = sample_length - waveform_length
                    padded_waveform = torch.nn.functional.pad(waveform, (0, padding_length), 'constant', 0)
                    segment = padded_waveform
                else:
                    # Randomly choose a starting point for the segment
                    start_idx = random.randint(0, waveform_length - sample_length)
                    segment = waveform[:, start_idx:start_idx + sample_length]

                # Append the encoded sample
                training_list.append([segment, label, encoded_SpeakerID, encoded_D_type, encoded_gender])

    # Print the encoding rules
    print("SpeakerID encoding:", speakerID_encoding)
    print("D_type encoding:", dtype_encoding)
    print("Gender encoding:", gender_encoding)

    return training_list

def generate_training_dataloader(args, data_list, target_length):
    # Separate the samples based on their labels
    label_0_samples = [s for s in data_list if s[1] == 0]
    label_1_samples = [s for s in data_list if s[1] == 1]

    # Find the minimum number of samples for balancing
    min_samples = min(len(label_0_samples), len(label_1_samples))
    print(len(label_0_samples))
    print(len(label_1_samples))
    data_per_class = min(target_length//2, min_samples) 

    # Randomly choose the same number of samples for each label
    balanced_samples_0 = random.sample(label_0_samples, min_samples)
    balanced_samples_1 = random.sample(label_1_samples, min_samples)

    # Combine the balanced samples
    balanced_samples = balanced_samples_0 + balanced_samples_1

    # Shuffle the balanced list to mix the samples
    random.shuffle(balanced_samples)
        
    # Instantiate the dataset
    dataset = CustomDataset(balanced_samples)

    # Create a DataLoader with the custom collate function
    dataloader = DataLoader(dataset, batch_size=args.bs, collate_fn=lambda batch: collate_fn(batch, args.wave_length))

    return dataloader

def get_dataset_version1():
    
    Dysarthric_sample_list_all, Healthy_sample_list_all = load_R01_from_folder()
    # Dysarthric_sample_list_all, Healthy_sample_list_all = load_R01_sentence_only_from_folder()
    ALS_sample_list_all = load_ALS_from_folder()
    user_sampling = generate_speaker_sampling_preset()

    i = 0

    # assigning speakers to different sets
    speaker_HC_list_train_for_DytDetec, speaker_HC_list_eval_for_DytDetec, speaker_HC_list_test_for_DytDetec = user_sampling[i][3]
    speaker_HC_list_train_for_ALSDetec, speaker_HC_list_eval_for_ALSDetec,speaker_HC_list_test_for_ALSDetec = user_sampling[i][2]
    Speaker_Dysarthric_list_train_all,Speaker_Dysarthric_list_eval_all, Speaker_Dysarthric_list_test_all = user_sampling[i][1]
    speaker_ALS_list_train_all,speaker_ALS_list_eval_all, speaker_ALS_list_test_all = user_sampling[i][0]
    # sorting the file lists via speakers
    Healthy_sample_ALSDetec_list_train_all,\
        Healthy_sample_ALSDetec_list_eval_all, \
            Healthy_sample_ALSDetec_list_test_all = split_list_by_participant_preset(Healthy_sample_list_all, speaker_HC_list_train_for_ALSDetec, speaker_HC_list_eval_for_ALSDetec, speaker_HC_list_test_for_ALSDetec)
    Healthy_sample_ALSDetec_list_train_all = load_wav_from_dir_list(Healthy_sample_ALSDetec_list_train_all)
    Healthy_sample_ALSDetec_list_eval_all = load_wav_from_dir_list(Healthy_sample_ALSDetec_list_eval_all)
    Healthy_sample_ALSDetec_list_test_all = load_wav_from_dir_list(Healthy_sample_ALSDetec_list_test_all)

    Healthy_sample_DytDetec_list_train_all, \
        Healthy_sample_DytDetec_list_eval_all, \
            Healthy_sample_DytDetec_list_test_all = split_list_by_participant_preset(Healthy_sample_list_all, speaker_HC_list_train_for_DytDetec, speaker_HC_list_eval_for_DytDetec, speaker_HC_list_test_for_DytDetec)
    Healthy_sample_DytDetec_list_train_all = load_wav_from_dir_list(Healthy_sample_DytDetec_list_train_all)
    Healthy_sample_DytDetec_list_eval_all = load_wav_from_dir_list(Healthy_sample_DytDetec_list_eval_all)
    Healthy_sample_DytDetec_list_test_all = load_wav_from_dir_list(Healthy_sample_DytDetec_list_test_all)

    ALS_sample_list_train_all, \
        ALS_sample_list_eval_all, \
            ALS_sample_list_test_all = split_list_by_participant_preset(ALS_sample_list_all, speaker_ALS_list_train_all,speaker_ALS_list_eval_all, speaker_ALS_list_test_all)
    ALS_sample_list_train_all = load_wav_from_dir_list(ALS_sample_list_train_all)
    ALS_sample_list_eval_all = load_wav_from_dir_list(ALS_sample_list_eval_all)
    ALS_sample_list_test_all = load_wav_from_dir_list(ALS_sample_list_test_all)

    Dyt_sample_list_train_all, \
        Dyt_sample_list_eval_all, \
            Dyt_sample_list_test_all = split_list_by_participant_preset(Dysarthric_sample_list_all, Speaker_Dysarthric_list_train_all,Speaker_Dysarthric_list_eval_all, Speaker_Dysarthric_list_test_all)
    Dyt_sample_list_train_all = load_wav_from_dir_list(Dyt_sample_list_train_all)
    Dyt_sample_list_eval_all = load_wav_from_dir_list(Dyt_sample_list_eval_all)
    Dyt_sample_list_test_all = load_wav_from_dir_list(Dyt_sample_list_test_all)

    # ALS_Detection_training_file_list = [ALS_sample_list_train_all, Healthy_sample_ALSDetec_list_train_all]
    # ALS_Detection_evaluation_file_list = [ALS_sample_list_eval_all, Healthy_sample_ALSDetec_list_eval_all]
    ALS_Detection_testing_file_list = [ALS_sample_list_test_all, Healthy_sample_ALSDetec_list_test_all]

    # Dyt_Detection_training_file_list = [Dyt_sample_list_train_all, Healthy_sample_DytDetec_list_train_all]
    # Dyt_Detection_evaluation_file_list = [Dyt_sample_list_eval_all, Healthy_sample_DytDetec_list_eval_all]
    Dyt_Detection_testing_file_list = [Dyt_sample_list_test_all, Healthy_sample_DytDetec_list_test_all]

    Training_file_list = [ALS_sample_list_train_all + Dyt_sample_list_train_all, Healthy_sample_ALSDetec_list_train_all + Healthy_sample_DytDetec_list_train_all]
    Evaluation_file_list = [ALS_sample_list_eval_all + Dyt_sample_list_eval_all, Healthy_sample_ALSDetec_list_eval_all + Healthy_sample_DytDetec_list_eval_all]

    Training_wav_list = create_training_set_from_list(Training_file_list)
    Evaluation_wav_list = create_training_set_from_list(Evaluation_file_list)
    ALS_detection_Testing_wav_list = create_training_set_from_list(ALS_Detection_testing_file_list)
    Dyt_detection_Testing_wav_list = create_training_set_from_list(Dyt_Detection_testing_file_list)

    return Training_wav_list, Evaluation_wav_list, ALS_detection_Testing_wav_list, Dyt_detection_Testing_wav_list

def get_dataset_version_sentenceOnly(args):
    Dysarthric_sample_list_all, Healthy_sample_list_all = load_R01_sentence_only_from_folder()
    ALS_sample_list_all = load_ALS_from_folder()
    user_sampling = generate_speaker_sampling_preset()

    i = 0

    # assigning speakers to different sets
    speaker_HC_list_train_for_DytDetec, speaker_HC_list_eval_for_DytDetec, speaker_HC_list_test_for_DytDetec = user_sampling[i][3]
    speaker_HC_list_train_for_ALSDetec, speaker_HC_list_eval_for_ALSDetec,speaker_HC_list_test_for_ALSDetec = user_sampling[i][2]
    Speaker_Dysarthric_list_train_all,Speaker_Dysarthric_list_eval_all, Speaker_Dysarthric_list_test_all = user_sampling[i][1]
    speaker_ALS_list_train_all,speaker_ALS_list_eval_all, speaker_ALS_list_test_all = user_sampling[i][0]
    # sorting the file lists via speakers
    Healthy_sample_ALSDetec_list_train_all,\
        Healthy_sample_ALSDetec_list_eval_all, \
            Healthy_sample_ALSDetec_list_test_all = split_list_by_participant_preset(Healthy_sample_list_all, speaker_HC_list_train_for_ALSDetec, speaker_HC_list_eval_for_ALSDetec, speaker_HC_list_test_for_ALSDetec)
    Healthy_sample_ALSDetec_list_train_all = load_wav_from_dir_list(Healthy_sample_ALSDetec_list_train_all)
    Healthy_sample_ALSDetec_list_eval_all = load_wav_from_dir_list(Healthy_sample_ALSDetec_list_eval_all)
    Healthy_sample_ALSDetec_list_test_all = load_wav_from_dir_list(Healthy_sample_ALSDetec_list_test_all)

    Healthy_sample_DytDetec_list_train_all, \
        Healthy_sample_DytDetec_list_eval_all, \
            Healthy_sample_DytDetec_list_test_all = split_list_by_participant_preset(Healthy_sample_list_all, speaker_HC_list_train_for_DytDetec, speaker_HC_list_eval_for_DytDetec, speaker_HC_list_test_for_DytDetec)
    Healthy_sample_DytDetec_list_train_all = load_wav_from_dir_list(Healthy_sample_DytDetec_list_train_all)
    Healthy_sample_DytDetec_list_eval_all = load_wav_from_dir_list(Healthy_sample_DytDetec_list_eval_all)
    Healthy_sample_DytDetec_list_test_all = load_wav_from_dir_list(Healthy_sample_DytDetec_list_test_all)

    ALS_sample_list_train_all, \
        ALS_sample_list_eval_all, \
            ALS_sample_list_test_all = split_list_by_participant_preset(ALS_sample_list_all, speaker_ALS_list_train_all,speaker_ALS_list_eval_all, speaker_ALS_list_test_all)
    ALS_sample_list_train_all = load_wav_from_dir_list(ALS_sample_list_train_all)
    ALS_sample_list_eval_all = load_wav_from_dir_list(ALS_sample_list_eval_all)
    ALS_sample_list_test_all = load_wav_from_dir_list(ALS_sample_list_test_all)

    Dyt_sample_list_train_all, \
        Dyt_sample_list_eval_all, \
            Dyt_sample_list_test_all = split_list_by_participant_preset(Dysarthric_sample_list_all, Speaker_Dysarthric_list_train_all,Speaker_Dysarthric_list_eval_all, Speaker_Dysarthric_list_test_all)
    Dyt_sample_list_train_all = load_wav_from_dir_list(Dyt_sample_list_train_all)
    Dyt_sample_list_eval_all = load_wav_from_dir_list(Dyt_sample_list_eval_all)
    Dyt_sample_list_test_all = load_wav_from_dir_list(Dyt_sample_list_test_all)

    # ALS_Detection_training_file_list = [ALS_sample_list_train_all, Healthy_sample_ALSDetec_list_train_all]
    # ALS_Detection_evaluation_file_list = [ALS_sample_list_eval_all, Healthy_sample_ALSDetec_list_eval_all]
    ALS_Detection_testing_file_list = [ALS_sample_list_test_all, Healthy_sample_ALSDetec_list_test_all]

    # Dyt_Detection_training_file_list = [Dyt_sample_list_train_all, Healthy_sample_DytDetec_list_train_all]
    # Dyt_Detection_evaluation_file_list = [Dyt_sample_list_eval_all, Healthy_sample_DytDetec_list_eval_all]
    Dyt_Detection_testing_file_list = [Dyt_sample_list_test_all, Healthy_sample_DytDetec_list_test_all]

    Training_file_list = [ALS_sample_list_train_all + Dyt_sample_list_train_all, Healthy_sample_ALSDetec_list_train_all + Healthy_sample_DytDetec_list_train_all]
    Evaluation_file_list = [ALS_sample_list_eval_all + Dyt_sample_list_eval_all, Healthy_sample_ALSDetec_list_eval_all + Healthy_sample_DytDetec_list_eval_all]

    Training_wav_list = create_training_set_from_list_sentence(Training_file_list, args.wave_length, args.sample_per_record)
    Evaluation_wav_list = create_training_set_from_list_sentence(Evaluation_file_list, args.wave_length, args.sample_per_record)
    ALS_detection_Testing_wav_list = create_training_set_from_list_sentence(ALS_Detection_testing_file_list, args.wave_length, args.sample_per_record)
    Dyt_detection_Testing_wav_list = create_training_set_from_list_sentence(Dyt_Detection_testing_file_list, args.wave_length, args.sample_per_record)

    return Training_wav_list, Evaluation_wav_list, ALS_detection_Testing_wav_list, Dyt_detection_Testing_wav_list