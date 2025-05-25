import os
import numpy as np

# import tensorflow as tf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pandas as pd
import matplotlib.pyplot as plt


def get_df_from_dir(target_dir, target_tag):
    event_file_list = os.listdir(target_dir)

    for file in event_file_list:
        if 'events' in file:
            event_file_name = file
    event_file_path = target_dir + '/' + event_file_name


    # Load the event file
    event_acc = EventAccumulator(event_file_path)
    event_acc.Reload()

    # Get available scalar tags
    scalar_tags = event_acc.Tags()['scalars']

    # Extract scalar data
    scalars = {}
    for tag in scalar_tags:
        events = event_acc.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        scalars[tag] = pd.DataFrame({'Step': steps, 'Value': values})

    # Plot the first available scalar

    # target_tag = 'eval/average_cls_loss'
    df = scalars[target_tag]

    return df

# Path to your event file
# event_file_path = "path/to/your/events.out.tfevents"
tag = 'eval/average_cls_loss'
save_path = "loss_curve.png"

target_dir = '/media/bearock/ssd_Speech/Dysarthria_FundationModel/ASR_MTL/exp_results/MTL_Normalized_E0_AllR01_50e_TGP_layerwise_scheduler_10_0p6_cls=2_e=50_bs=4_ctcW=0.1_2025-01-15_18:18:58'
df_TGP_MTL_asr = get_df_from_dir(target_dir,tag)

target_dir = '/media/bearock/ssd_Speech/Dysarthria_FundationModel/ASR_MTL/exp_results/MTL_Normalized_E0_AllR01_25e_TGP_layerwise_scheduler_10_0p6_CLSMain_cls=2_e=50_bs=4_ctcW=0.1_2025-01-16_11:10:58'
df_TGP_MTL_CLS = get_df_from_dir(target_dir,tag)

target_dir = '/media/bearock/ssd_Speech/Dysarthria_FundationModel/ASR_Sample/exp_results/_cls=2_e=50_bs=4_ctcW=0.0_2025-01-17_12:57:44'
df_STL = get_df_from_dir(target_dir,tag)

target_dir = '/media/bearock/ssd_Speech/Dysarthria_FundationModel/ASR_Sample/exp_results/_cls=2_e=50_bs=4_ctcW=0.1_2025-01-17_04:15:19'
df_STD_MTL = get_df_from_dir(target_dir,tag)

plt.figure(figsize=(10, 5))
plt.plot(df_TGP_MTL_asr['Step'], df_TGP_MTL_asr['Value'], label='TGP MTL, using ASR as main task')
plt.plot(df_TGP_MTL_CLS['Step'], df_TGP_MTL_CLS['Value'], label='TGP MTL, using Detection as main task')
plt.plot(df_STL['Step'], df_STL['Value'], label='Baseline: STL')
plt.plot(df_STD_MTL['Step'], df_STD_MTL['Value'], label='Baseline: Standard MTL')

plt.xlabel("Step")
plt.ylabel("Value")
# plt.title(f"TensorBoard Scalar: {tag}")
plt.legend()
plt.show()
plt.savefig(save_path, dpi=300, bbox_inches='tight')
