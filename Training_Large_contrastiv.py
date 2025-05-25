
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


    # train_naive_multitask(args,log_file_dir,visualization_save_dir, model, optimizer, criterion_d,criterion_spe, R01_processed_list_train_loader,SpeakerEmbedding_processed_list_train_loader,ALS_test_loader)

def train_contrastive(args,log_file_dir, Training_wav_list, Evaluation_wav_list, model, optimizer, criterion):

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

        for (data_d, target_d) in training_loader:

            optimizer.zero_grad()

            data_d = data_d.squeeze().cuda()
            target_d = target_d.unsqueeze(-1).float().cuda()

            output_d,output_f = model(data_d)
            print(output_f.shape)

            loss = criterion(output_d, target_d)

            loss.backward()
            loss_acc += loss.item()/len(data_d)
            optimizer.step()
        scheduler.step()
        loss_acc/=len(training_loader)

        # if epoch>0 and epoch%5 == 0:
        _, test_loss = test_by_dataloader(epoch,model,criterion, evaluation_loader)

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

def test_by_dataloader(epoch,model,criterion, dataloader):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    # Disable gradient calculation for inference
    with torch.no_grad():
        for batch_data, batch_labels in dataloader:
            # Move data and labels to the specified device
            batch_data, batch_labels = batch_data.cuda(), batch_labels.cuda()
            batch_data = batch_data.squeeze()
            batch_labels = batch_labels.unsqueeze(-1).float()

            # Forward pass to get predictions
            outputs,_ = model(batch_data)

            # Calculate the loss
            loss = criterion(outputs, batch_labels)
            total_loss += loss.item()/len(batch_data)

    # Calculate average loss and accuracy
    average_loss = total_loss / len(dataloader)
    accuracy = 0

    return accuracy,average_loss

def main_randomly_chosen():
    args = args_parser()
    print(args.bs)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 1  # Example: classification problem with 10 classes
    model = Wav2Vec2Classifier(num_classes).to(device)

    Training_wav_list, Evaluation_wav_list, Testing_wav_list = get_dataset_large_version1()

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # criterion = ContrastiveLoss()
    criterion = nn.BCEWithLogitsLoss()
    log_file_dir = args.save_dir + '/running_log.csv'

    train_contrastive(args,log_file_dir, Training_wav_list, Evaluation_wav_list, model, optimizer, criterion)
    
if __name__ == "__main__":
    main_randomly_chosen()
