from dataset_dir import *
from Model_fundation import *
from utils import *

import csv
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence



def add_sample_to_data(data, input_list, category, split):
    for sample in input_list:
        FILE_INDEX = sample[1]+'_' + sample[4]
        file_path = sample[0]
        text = sample[5]
        csv_row = [FILE_INDEX, file_path, category, text, split]
        data.append(csv_row)
    return data

def add_sample_to_data_with_label(data, input_list, split):
    for sample in input_list:
        FILE_INDEX = sample[1]+'_' + sample[4]
        file_path = sample[0]
        text = sample[5]
        category = sample[2]
        csv_row = [FILE_INDEX, file_path, category, text, split]
        data.append(csv_row)
    return data

def encode_classes_with_counts_and_limit(data):
    """
    Encode the classes into numeric indices, record sample counts, 
    and limit the number of samples for each class.

    Parameters:
        data (list of lists): Each item is a list where the third value is the class.
        max_samples_per_class (int): Maximum number of samples allowed per class.

    Returns:
        tuple: (limited_data, class_to_index, class_counts)
            - limited_data: List with class values replaced by their numeric indices,
                            and limited to max_samples_per_class per class.
            - class_to_index: Dictionary mapping class names to their indices.
            - class_counts: Dictionary recording the count of samples for each class.
    """
    class_to_index = {'AD': 0, 'CP': 1, 'EP': 2, 'SD': 3, 'PD': 4}
    class_counts = {'AD': 0, 'CP': 0, 'EP': 0, 'SD': 0, 'PD': 0}
    class_items = {'AD': [], 'CP': [], 'EP': [], 'SD': [], 'PD': []}

    # Group items by class and assign indices
    for item in data:
        class_name = item[2]  # The class is the third value in each item
        if class_name == 'MS': continue
        class_index = class_to_index[class_name]
        item[2] = class_index
        class_items[class_name].append(item)
        class_counts[class_name] += 1

    # Determine the size of the second-largest class
    sorted_counts = sorted(class_counts.values(), reverse=True)
    if len(sorted_counts) > 1:
        max_samples_per_class = sorted_counts[1]
    else:
        max_samples_per_class = sorted_counts[0]  # If only one class exists

    # Limit samples per class
    limited_data = []
    for class_index, items in class_items.items():
        if len(items) > max_samples_per_class:
            limited_data.extend(random.sample(items, max_samples_per_class))
        else:
            limited_data.extend(items)
        class_counts[class_index] = min(class_counts[class_index], max_samples_per_class)

    return limited_data, class_to_index, class_counts

# Dysarthric_sample_list_all, Healthy_sample_list_all = load_R01_sentence_only_from_folder()
Dysarthric_sample_list_all, Healthy_sample_list_all = load_R01_from_folder()
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

print('Healthy_sample_ALSDetec_list_train_all:', len(Healthy_sample_ALSDetec_list_train_all))
print('Healthy_sample_ALSDetec_list_eval_all:', len(Healthy_sample_ALSDetec_list_eval_all))
print('Healthy_sample_ALSDetec_list_test_all:', len(Healthy_sample_ALSDetec_list_test_all))

Healthy_sample_DytDetec_list_train_all, \
    Healthy_sample_DytDetec_list_eval_all, \
        Healthy_sample_DytDetec_list_test_all = split_list_by_participant_preset(Healthy_sample_list_all, speaker_HC_list_train_for_DytDetec, speaker_HC_list_eval_for_DytDetec, speaker_HC_list_test_for_DytDetec)

print('Healthy_sample_DytDetec_list_train_all:', len(Healthy_sample_DytDetec_list_train_all))
print('Healthy_sample_DytDetec_list_eval_all:', len(Healthy_sample_DytDetec_list_eval_all))
print('Healthy_sample_DytDetec_list_test_all:', len(Healthy_sample_DytDetec_list_test_all))

ALS_sample_list_train_all, \
    ALS_sample_list_eval_all, \
        ALS_sample_list_test_all = split_list_by_participant_preset(ALS_sample_list_all, speaker_ALS_list_train_all,speaker_ALS_list_eval_all, speaker_ALS_list_test_all)

print('ALS_sample_list_train_all:', len(ALS_sample_list_train_all))
print('ALS_sample_list_eval_all:', len(ALS_sample_list_eval_all))
print('ALS_sample_list_test_all:', len(ALS_sample_list_test_all))

Dyt_sample_list_train_all, \
    Dyt_sample_list_eval_all, \
        Dyt_sample_list_test_all = split_list_by_participant_preset(Dysarthric_sample_list_all, Speaker_Dysarthric_list_train_all,Speaker_Dysarthric_list_eval_all, Speaker_Dysarthric_list_test_all)
print('Dyt_sample_list_train_all:', len(Dyt_sample_list_train_all))
print('Dyt_sample_list_eval_all:', len(Dyt_sample_list_eval_all))
print('Dyt_sample_list_test_all:', len(Dyt_sample_list_test_all))

# data = []
# data = add_sample_to_data(data, ALS_sample_list_train_all, 1, 'test')

# data_healthy = []
# data_healthy = add_sample_to_data(data_healthy, Healthy_sample_ALSDetec_list_train_all, 0, 'test')
# balanced_data_healthy = random.sample(data_healthy, len(data))
# data += balanced_data_healthy

#  = Dyt_sample_list_train_all
encoded_data_training, class_to_index_training, class_counts_training = encode_classes_with_counts_and_limit(Dyt_sample_list_train_all)
encoded_data_evaluation, class_to_index_evaluation, class_counts_evaluation = encode_classes_with_counts_and_limit(Dyt_sample_list_eval_all)
encoded_data_testing, class_to_index_testing, class_counts_testing = encode_classes_with_counts_and_limit(Dyt_sample_list_test_all)
print("===================Train========================")
print("\nClass to Index Mapping:")
print(class_to_index_training)

print("\nClass Counts:")
print(class_counts_training)

print("===================eval========================")
print("\nClass to Index Mapping:")
print(class_to_index_evaluation)

print("\nClass Counts:")
print(class_counts_evaluation)

print("===================Test========================")
print("\nClass to Index Mapping:")
print(class_to_index_testing)

print("\nClass Counts:")
print(class_counts_testing)

data = []
data = add_sample_to_data_with_label(data, encoded_data_training, 'train')
data = add_sample_to_data_with_label(data, encoded_data_evaluation, 'valid')
data = add_sample_to_data_with_label(data, encoded_data_testing, 'test')

# Specify the header
header = ["name", "path", "category", "text", "split"]

# Write the data to a CSV file
with open("dataset_classification_R01.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    # Write the header
    writer.writerow(header)
    # Write the data
    writer.writerows(data)

print("CSV file created successfully!")
