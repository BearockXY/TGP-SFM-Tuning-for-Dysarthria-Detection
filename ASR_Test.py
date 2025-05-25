from dataset_dir import *
from Model_fundation import *
from utils import *

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

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

Healthy_sample_DytDetec_list_train_all, \
    Healthy_sample_DytDetec_list_eval_all, \
        Healthy_sample_DytDetec_list_test_all = split_list_by_participant_preset(Healthy_sample_list_all, speaker_HC_list_train_for_DytDetec, speaker_HC_list_eval_for_DytDetec, speaker_HC_list_test_for_DytDetec)

ALS_sample_list_train_all, \
    ALS_sample_list_eval_all, \
        ALS_sample_list_test_all = split_list_by_participant_preset(ALS_sample_list_all, speaker_ALS_list_train_all,speaker_ALS_list_eval_all, speaker_ALS_list_test_all)

Dyt_sample_list_train_all, \
    Dyt_sample_list_eval_all, \
        Dyt_sample_list_test_all = split_list_by_participant_preset(Dysarthric_sample_list_all, Speaker_Dysarthric_list_train_all,Speaker_Dysarthric_list_eval_all, Speaker_Dysarthric_list_test_all)

# # ALS_Detection_training_file_list = [ALS_sample_list_train_all, Healthy_sample_ALSDetec_list_train_all]
# # ALS_Detection_evaluation_file_list = [ALS_sample_list_eval_all, Healthy_sample_ALSDetec_list_eval_all]
# ALS_Detection_testing_file_list = [ALS_sample_list_test_all, Healthy_sample_ALSDetec_list_test_all]

# # Dyt_Detection_training_file_list = [Dyt_sample_list_train_all, Healthy_sample_DytDetec_list_train_all]
# # Dyt_Detection_evaluation_file_list = [Dyt_sample_list_eval_all, Healthy_sample_DytDetec_list_eval_all]
# Dyt_Detection_testing_file_list = [Dyt_sample_list_test_all, Healthy_sample_DytDetec_list_test_all]

# Training_file_list = [ALS_sample_list_train_all + Dyt_sample_list_train_all, Healthy_sample_ALSDetec_list_train_all + Healthy_sample_DytDetec_list_train_all]
# Evaluation_file_list = [ALS_sample_list_eval_all + Dyt_sample_list_eval_all, Healthy_sample_ALSDetec_list_eval_all + Healthy_sample_DytDetec_list_eval_all]






import torch
import torchaudio

print(torch.__version__)
print(torchaudio.__version__)

torch.random.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)

import IPython
import matplotlib.pyplot as plt
from torchaudio.utils import download_asset

SPEECH_FILE = '/media/bearock/ssd_Speech/data/R01_dataset/Dysarthric_Data_Final/ADF19_MIP_habitual_sentences/ADF19_MIP_ABC_normal_In this famous coffee shop they serve the best doughnuts in town_Spastic.wav'

bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H

print("Sample Rate:", bundle.sample_rate)

print("Labels:", bundle.get_labels())

model = bundle.get_model().to(device)

print(model.__class__)

waveform, sample_rate = torchaudio.load(SPEECH_FILE)
waveform = waveform.to(device)

if sample_rate != bundle.sample_rate:
    waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)

with torch.inference_mode():
    features, _ = model.extract_features(waveform)

fig, ax = plt.subplots(len(features), 1, figsize=(16, 4.3 * len(features)))
for i, feats in enumerate(features):
    ax[i].imshow(feats[0].cpu(), interpolation="nearest")
    ax[i].set_title(f"Feature from transformer layer {i+1}")
    ax[i].set_xlabel("Feature dimension")
    ax[i].set_ylabel("Frame (time-axis)")
fig.tight_layout()

with torch.inference_mode():
    emission, _ = model(waveform)

plt.imshow(emission[0].cpu().T, interpolation="nearest")
plt.title("Classification result")
plt.xlabel("Frame (time-axis)")
plt.ylabel("Class")
plt.tight_layout()
print("Class labels:", bundle.get_labels())

class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, labels, blank=0):
        super().__init__()
        self.labels = labels
        self.blank = blank

    def forward(self, emission: torch.Tensor) -> str:
        """Given a sequence emission over labels, get the best path string
        Args:
          emission (Tensor): Logit tensors. Shape `[num_seq, num_label]`.

        Returns:
          str: The resulting transcript
        """
        indices = torch.argmax(emission, dim=-1)  # [num_seq,]
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i for i in indices if i != self.blank]
        return "".join([self.labels[i] for i in indices])
    

decoder = GreedyCTCDecoder(labels=bundle.get_labels())
transcript = decoder(emission[0])
print(transcript)