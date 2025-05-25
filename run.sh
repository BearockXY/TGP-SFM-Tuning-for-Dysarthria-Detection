#!/bin/bash


# python MTL_ALS_Dyth_contrastive.py --lr 0.0001 --skipping_threthold 0.1 --save_dir './test_result/contrastive_learning_transformer/TGP_MTL_ALSDetect_DYSSpe/SPE_pretrained/TS10000_lrem3_skip_0p1'
# python MTL_ALS_Dyth_contrastive.py --lr 0.0005 --skipping_threthold 0.05 --save_dir './test_result/contrastive_learning_transformer/TGP_MTL_ALSDetect_DYSSpe/SPE_pretrained/TS10000_5rem3_skip_0p05'
# python MTL_ALS_Dyth_contrastive.py --lr 0.001 --skipping_threthold 0.1 --save_dir './test_result/contrastive_learning_transformer/TGP_MTL_ALSDetect_DYSSpe/SPE_pretrained/TS10000_lrem2_skip_0p1'
# python MTL_ALS_Dyth_contrastive.py --lr 0.005 --skipping_threthold 0.1 --save_dir './test_result/contrastive_learning_transformer/TGP_MTL_ALSDetect_DYSSpe/SPE_pretrained/TS10000_5rem3_skip_0p1'
# /media/bearock/ssd_Speech/ALS_R01/test_result/ALS_detection_from_Dyt/Baseline/MTL_naive


# python MTL_ALS_Dyth_contrastive.py --lr 0.0005 --gamma 1.0 --skipping_threthold 0.1 --save_dir './test_result/contrastive_learning_transformer/TGP_MTL_ALSDetect_DYSSpe/SPE_pretrained_weightsaving/TS10000_5rem3_skip_0p1_gamma10'
# python MTL_ALS_Dyth_contrastive.py --lr 0.0005 --gamma 1.0 --skipping_threthold 0.05 --save_dir './test_result/contrastive_learning_transformer/TGP_MTL_ALSDetect_DYSSpe/SPE_pretrained_weightsaving/TS10000_5rem3_skip_0p05_gamma10'
# python MTL_ALS_Dyth_contrastive.py --lr 0.0005 --gamma 1.0 --skipping_threthold 0.0 --save_dir './test_result/contrastive_learning_transformer/TGP_MTL_ALSDetect_DYSSpe/SPE_pretrained_weightsaving/TS10000_5rem3_skip_0p0_gamma10'
# Define a list of argument values

# lr_values=(0.005 0.001 0.0005 0.0001 0.00005)
save_dir='./test_result/MTL_SPE_DYD_sentenceOnly_end_to_end_Tune'
# vis_dir_org='./test_result_weight/SPE_tuning/parameter_'
# Loop through each argument value


# for lr in "${lr_values[@]}"
# do

#     # Run the Python script with the current argument value
#     save_dir="$save_dir_org$'lr'$lr"
#     vis_dir="$vis_dir_org$'lr'$lr"
#     python Training_Stage_SpeakerEmbedding.py --lr $lr --save_dir $save_dir --umap_save_dir $vis_dir

# done

# 
python Training_STL_contrastive.py --save_dir $save_dir --lr 0.00001