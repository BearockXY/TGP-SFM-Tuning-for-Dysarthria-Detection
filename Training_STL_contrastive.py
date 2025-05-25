
import torch
import torch.nn as nn
import torchaudio
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import random
import csv
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

from options import args_parser
from dataset_dir import *
from Model_fundation import *
from utils import *
from data_preparing import *

from R01_large_loader import get_dataset_large_version1

class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        features = F.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features, features.T) / self.temperature

        # Exclude self-comparisons by subtracting the identity matrix
        logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        logits = similarity_matrix - logits_max.detach()
        
        # Apply mask to select only positives for each anchor
        mask = mask * (1 - torch.eye(mask.size(0)).to(device))
        exp_logits = torch.exp(logits) * mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-9)

        # Loss calculation
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-9)
        loss = -mean_log_prob_pos.mean()

        return loss

def train_naive_multitask(args,log_file_dir, Training_wav_list, Evaluation_wav_list, model, optimizer, criterion1, criterion2):

    eval_optimal = 10000
    testing_acc = 0
    epoch_optimal = 0
    training_length = 4400
    evaluation_length = 700
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)  # Adjust parameters as needed

    logfile = open(log_file_dir, 'w')
    logwriter = csv.DictWriter(logfile, fieldnames=['epoch', 'Train_loss', 'Train_loss_d', 'Train_loss_SPE', 'Eval_loss', 'Eval_loss_d', 'Eval_loss_SPE'])
    logwriter.writeheader()
    
    for epoch in tqdm(range(args.epochs)):
        model.train()
        loss_acc_spe = 0
        loss_acc_d = 0
        loss_acc = 0
        training_loader = generate_training_dataloader(args, Training_wav_list, training_length)
        evaluation_loader = generate_training_dataloader(args, Evaluation_wav_list, evaluation_length)


        # segment, label, encoded_SpeakerID, encoded_D_type, encoded_gender
        for (data_d, target_d, target_SPE, _, _ ) in training_loader:

            optimizer.zero_grad()

            data_d = data_d.squeeze().cuda()
            target_d = target_d.unsqueeze(-1).float().cuda()
            target_SPE = target_SPE.unsqueeze(-1).float().cuda()

            output_d,output_f = model(data_d)
            # print(output_f.shape)

            loss_d = criterion1(output_d,target_d) 
            loss_spe = criterion2(output_f, target_SPE)
            loss = loss_d + loss_spe

            loss.backward()

            loss_acc_d += loss_d.item()/len(data_d)
            loss_acc_spe += loss_spe.item()/len(data_d)
            loss_acc += loss.item()/len(data_d)

            optimizer.step()
        scheduler.step()
        loss_acc_d/=len(training_loader)
        loss_acc_spe/=len(training_loader)
        loss_acc/=len(training_loader)

        # if epoch>0 and epoch%5 == 0:
        _, test_loss,test_loss_d,test_loss_SPE = test_by_dataloader_MTL(args, epoch,model, criterion1 ,criterion2, evaluation_loader)

        if test_loss < eval_optimal:
            eval_optimal = test_loss
            epoch_optimal = epoch
            weight_name = args.save_dir + '/weight_epoch_'+str(epoch)+'.pt'
            torch.save(model.state_dict(), weight_name)
            
        # if loss_acc < loss_optimal:

            # last_weights = None
        print("==================runtime evaluation results=====================")
        print("epoch ",epoch)
        print("training loss = ",loss_acc)
        print("evaluation loss = ", test_loss)
        logwriter.writerow(dict(epoch=epoch, Train_loss=loss_acc,Train_loss_d=loss_acc_d,Train_loss_SPE=loss_acc_spe, Eval_loss=test_loss, Eval_loss_d=test_loss_d, Eval_loss_SPE=test_loss_SPE))

        # logwriter.writerow(dict(epoch=epoch, Train_loss_d=loss_acc_d,Train_loss_spe=loss_acc_spe,Eval_loss_d=eval_loss,eval_acc=eval_accuracy,Test_loss_d=test_loss, test_acc=test_accuracy))
    # print("==================Done Training, evaluation results=====================")
    # print("optimal evaluation accuracy so far is ",eval_optimal, " at epoch ",epoch_optimal, "testing accuracy is ",testing_acc )
    # logfile.close()
    #     

def train_naive_simpletask(args,log_file_dir, Training_wav_list, Evaluation_wav_list, model, optimizer, criterion):

    eval_optimal = 10000
    testing_acc = 0
    epoch_optimal = 0
    training_length = 4400
    evaluation_length = 700
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)  # Adjust parameters as needed

    logfile = open(log_file_dir, 'w')
    logwriter = csv.DictWriter(logfile, fieldnames=['epoch', 'Train_loss', 'Eval_loss'])
    logwriter.writeheader()
    
    for epoch in tqdm(range(args.epochs)):
        model.train()
        loss = 0
        
        loss_acc = 0
        training_loader = generate_training_dataloader(args, Training_wav_list, training_length)
        evaluation_loader = generate_training_dataloader(args, Evaluation_wav_list, evaluation_length)


        # segment, label, encoded_SpeakerID, encoded_D_type, encoded_gender
        for (data_d, _, target_SPE, _, _ ) in training_loader:

            optimizer.zero_grad()

            data_d = data_d.squeeze().cuda()
            target_SPE = target_SPE.unsqueeze(-1).float().cuda()

            output_d,output_f = model(data_d)
            # print(output_f.shape)

            loss = criterion(output_f, target_SPE)

            loss.backward()
            loss_acc += loss.item()/len(data_d)
            optimizer.step()
        scheduler.step()
        loss_acc/=len(training_loader)

        # if epoch>0 and epoch%5 == 0:
        _, test_loss = test_by_dataloader(args, epoch,model,criterion, evaluation_loader)

        if test_loss < eval_optimal:
            eval_optimal = test_loss
            epoch_optimal = epoch
            weight_name = args.save_dir + '/weight_epoch_'+str(epoch)+'.pt'
            torch.save(model.state_dict(), weight_name)
            
        # if loss_acc < loss_optimal:

            # last_weights = None
        print("==================runtime evaluation results=====================")
        print("epoch ",epoch)
        print("training loss = ",loss_acc)
        print("evaluation loss = ", test_loss)
        logwriter.writerow(dict(epoch=epoch, Train_loss=loss_acc, Eval_loss=test_loss))

        # logwriter.writerow(dict(epoch=epoch, Train_loss_d=loss_acc_d,Train_loss_spe=loss_acc_spe,Eval_loss_d=eval_loss,eval_acc=eval_accuracy,Test_loss_d=test_loss, test_acc=test_accuracy))
    # print("==================Done Training, evaluation results=====================")
    # print("optimal evaluation accuracy so far is ",eval_optimal, " at epoch ",epoch_optimal, "testing accuracy is ",testing_acc )
    # logfile.close()

def test_by_dataloader_MTL(args, epoch,model,criterion1, criterion2, dataloader):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    total_loss_spe = 0.0
    total_loss_d = 0.0
    # Disable gradient calculation for inference
    visualization_save_dir = args.save_dir + '/UMAP_by_epoch'
    if not os.path.exists(visualization_save_dir):
        os.makedirs(visualization_save_dir)
    out_feature = []
    out_label = []
    out_SpeakerID = []
    out_D_type = []
    out_gender = []

    with torch.no_grad():
        for batch_data, batch_labels, batch_SpeakerID, batch_D_type, batch_gender in dataloader:
            # Move data and labels to the specified device
            batch_data, batch_labels, batch_SpeakerID = batch_data.cuda(),batch_labels.cuda(), batch_SpeakerID.cuda()
            batch_data = batch_data.squeeze()

            batch_labels = batch_labels.unsqueeze(-1).float()
            batch_SpeakerID = batch_SpeakerID.unsqueeze(-1).float()
            batch_D_type = batch_D_type.unsqueeze(-1).float()
            batch_gender = batch_gender.unsqueeze(-1).float()

            # Forward pass to get predictions
            outputs,Umap_feature = model(batch_data)

            out_feature.append(Umap_feature.cpu().detach().numpy())
            # print(batch_labels)
            # print(batch_SpeakerID)

            # Calculate the loss
            loss_d = criterion1(outputs, batch_labels)
            loss_spe = criterion2(Umap_feature, batch_SpeakerID)
            total_loss_spe += loss_spe.item()/len(batch_data)
            total_loss_d += loss_d.item()/len(batch_data)

            out_label.append(batch_labels.squeeze().cpu().numpy())
            out_SpeakerID.append(batch_SpeakerID.squeeze().cpu().numpy())
            out_D_type.append(batch_D_type.squeeze().cpu().numpy())
            out_gender.append(batch_gender.squeeze().cpu().numpy())

            loss = loss_d + loss_spe
            total_loss += loss.item()/len(batch_data)
    
    # print(len(out_feature))
    # print(len(out_label))
    # print(out_label)
    # Calculate average loss and accuracy
    average_loss = total_loss / len(dataloader)
    total_loss_spe /= len(dataloader)
    total_loss_d /= len(dataloader)
    accuracy = 0

    random_int = random.randint(0, 100)
    if not any(np.isnan(arr).any() for arr in out_feature):

        umap_title = visualization_save_dir + "/umap_total_at_epoch_" + str(epoch) +'.jpg'
        make_tSNE(out_feature,out_label,umap_title, average_loss, random_int)

        umap_title = visualization_save_dir + "/umap_SpeakerID_at_epoch_" + str(epoch) +'.jpg'
        make_tSNE(out_feature,out_SpeakerID,umap_title, average_loss, random_int)

        pairs = [(i, j) for i in range(18) for j in range(18)]
        visualization_save_dir_pairwise = visualization_save_dir + '/Speaker_pair'
        if not os.path.exists(visualization_save_dir_pairwise):
            os.makedirs(visualization_save_dir_pairwise)
        for pair in pairs:
            if pair[0] == pair[1]: continue
            random_int_2 = random.randint(0, 100)
            umap_title = visualization_save_dir_pairwise + "/umap_SpeakerIDPair" +str(pair[0])+'_'+str(pair[1])+ "_at_epoch_" + str(epoch) +'.jpg'
            # make_tSNE_filtered(out_feature,out_SpeakerID,umap_title, average_loss, random_int_2, pair)
            make_tSNE_filtered_V2(out_feature,out_SpeakerID,umap_title, average_loss, random_int_2, pair)

        umap_title = visualization_save_dir + "/umap_DType_at_epoch_" + str(epoch) +'.jpg'
        make_tSNE(out_feature,out_D_type,umap_title, average_loss, random_int)

        umap_title = visualization_save_dir + "/umap_gender_at_epoch_" + str(epoch) +'.jpg'
        make_tSNE(out_feature,out_gender,umap_title, average_loss, random_int)

    return accuracy,average_loss,total_loss_d, total_loss_spe

def test_by_dataloader(args, epoch,model,criterion, dataloader):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    # Disable gradient calculation for inference
    visualization_save_dir = args.save_dir + '/UMAP_by_epoch'
    if not os.path.exists(visualization_save_dir):
        os.makedirs(visualization_save_dir)
    out_feature = []
    out_label = []
    out_SpeakerID = []
    out_D_type = []
    out_gender = []

    with torch.no_grad():
        for batch_data, batch_labels, batch_SpeakerID, batch_D_type, batch_gender in dataloader:
            # Move data and labels to the specified device
            batch_data, batch_SpeakerID = batch_data.cuda(), batch_SpeakerID.cuda()
            batch_data = batch_data.squeeze()

            batch_SpeakerID = batch_SpeakerID.unsqueeze(-1).float()

            batch_SpeakerID = batch_SpeakerID.unsqueeze(-1).float()
            batch_D_type = batch_D_type.unsqueeze(-1).float()
            batch_gender = batch_gender.unsqueeze(-1).float()

            # Forward pass to get predictions
            outputs,Umap_feature = model(batch_data)

            out_feature.append(Umap_feature.cpu().detach().numpy())
            # print(batch_labels)
            # print(batch_SpeakerID)

            out_label.append(batch_labels.squeeze().cpu().numpy())
            out_SpeakerID.append(batch_SpeakerID.squeeze().cpu().numpy())
            out_D_type.append(batch_D_type.squeeze().cpu().numpy())
            out_gender.append(batch_gender.squeeze().cpu().numpy())

            # Calculate the loss
            loss = criterion(Umap_feature, batch_SpeakerID)
            total_loss += loss.item()/len(batch_data)
    
    # print(len(out_feature))
    # print(len(out_label))
    # print(out_label)
    # Calculate average loss and accuracy
    average_loss = total_loss / len(dataloader)
    accuracy = 0
    random_int = random.randint(0, 100)
    if not any(np.isnan(arr).any() for arr in out_feature):

        umap_title = visualization_save_dir + "/umap_total_at_epoch_" + str(epoch) +'.jpg'
        make_tSNE(out_feature,out_label,umap_title, average_loss, random_int)

        umap_title = visualization_save_dir + "/umap_SpeakerID_at_epoch_" + str(epoch) +'.jpg'
        make_tSNE(out_feature,out_SpeakerID,umap_title, average_loss, random_int)

        pairs = [(i, j) for i in range(18) for j in range(18)]
        visualization_save_dir_pairwise = visualization_save_dir + '/Speaker_pair'
        if not os.path.exists(visualization_save_dir_pairwise):
            os.makedirs(visualization_save_dir_pairwise)
        for pair in pairs:
            if pair[0] == pair[1]: continue
            random_int_2 = random.randint(0, 100)
            umap_title = visualization_save_dir_pairwise + "/umap_SpeakerIDPair" +str(pair[0])+'_'+str(pair[1])+ "_at_epoch_" + str(epoch) +'.jpg'
            # make_tSNE_filtered(out_feature,out_SpeakerID,umap_title, average_loss, random_int_2, pair)
            make_tSNE_filtered_V2(out_feature,out_SpeakerID,umap_title, average_loss, random_int_2, pair)

        umap_title = visualization_save_dir + "/umap_DType_at_epoch_" + str(epoch) +'.jpg'
        make_tSNE(out_feature,out_D_type,umap_title, average_loss, random_int)

        umap_title = visualization_save_dir + "/umap_gender_at_epoch_" + str(epoch) +'.jpg'
        make_tSNE(out_feature,out_gender,umap_title, average_loss, random_int)

    return accuracy,average_loss

def main_randomly_chosen():
    args = args_parser()
    print(args.bs)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 1  # Example: classification problem with 10 classes
    model = Wav2Vec2Classifier(num_classes).to(device)

    # Training_wav_list, _, ALS_detection_Testing_wav_list, Dyt_detection_Testing_wav_list = get_dataset_version1()
    Training_wav_list, Evaluation_wav_list, ALS_detection_Testing_wav_list, Dyt_detection_Testing_wav_list = get_dataset_version_sentenceOnly(args)
    # Training_wav_list, _, Testing_wav_list = get_dataset_large_version1()

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # criterion = ContrastiveLoss()
    bce_loss_fn = nn.BCEWithLogitsLoss().to(device)
    contrastive_loss_fn = SupervisedContrastiveLoss(temperature=0.5).to(device)
    log_file_dir = args.save_dir + '/running_log.csv'

    # train_naive_simpletask(args,log_file_dir, Training_wav_list, Evaluation_wav_list, model, optimizer, criterion)
    train_naive_multitask(args,log_file_dir, Training_wav_list, Evaluation_wav_list, model, optimizer, bce_loss_fn,contrastive_loss_fn)
    
if __name__ == "__main__":
    main_randomly_chosen()
