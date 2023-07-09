
import os
import shutil
import time
import datetime
from tkinter import X
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchvision.models import densenet121
import torchvision.transforms as transforms
# import EarlyStopping
from pytorchtools import EarlyStopping
from PIL import Image
import copy
import numpy as np
import pandas as pd
import seaborn as sn

from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer
Image.MAX_IMAGE_PIXELS = None
# tensorboard log save
log_dir = "./DDH_log_dir"

class MyConfig:
    data_root = os.path.join(os.getcwd(),'datasets')
    train_dir = 'train'
    valid_dir = 'valid'
    test_dir = 'test'
    category_list = ['101','201','202','203','204','205','206','207','301','302','303']
    now = datetime.datetime.now().strftime('%y%m%d')
    time = datetime.datetime.now().strftime('%H%M')
    MODEL_PATH = os.path.join(os.getcwd(), 'model', 'NIA_ship', now+'_'+time)
    txt_log_dir = os.path.join(os.getcwd(), 'log')
    LOGGING_FILE = os.path.join(txt_log_dir, now+'_'+time+'_NIA_ship_training_log.txt')
    GPU_NUM = 0

    IMG_SIZE = (224, 224)
    my_transforms = transforms.Compose([           
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
    ])
    transforms_no_resize = transforms.Compose([                  
        transforms.ToTensor(),
        # transforms.Normalize(IMG_MEAN, IMG_STD)
    ])
        
    criterion = nn.CrossEntropyLoss()
    num_workers = 4 
    batch_size = 32
    lr = 1e-4
    n_epochs = 50


def make_dirs():
    if not os.path.exists(log_dir):
        os.makedirs(log_dir) 
    if not os.path.exists(MyConfig.MODEL_PATH):
        os.makedirs(MyConfig.MODEL_PATH)
    if not os.path.exists(MyConfig.txt_log_dir):
        os.makedirs(MyConfig.txt_log_dir)
    if os.path.exists(MyConfig.LOGGING_FILE):
        os.remove(MyConfig.LOGGING_FILE)
        print('?')


# https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
  

def making_data_list(dataset):
    image_root = os.path.join(MyConfig.data_root, dataset)
    image_list = os.listdir(image_root)
    data_list = []

    for image in image_list:
        image_path = os.path.join(image_root, image)
        data_list.append(image_path)

    return data_list


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data_list, transform=None):
        super(MyDataset, self).__init__()
        self.data_list = data_list
        self.transform = transform

    def __getitem__(self, index):
        try:
            img_path = self.data_list[index]  
            img = Image.open(img_path).convert('RGB')

            if self.transform is not None:
                img = self.transform(img)

            label = img_path.split(os.path.sep)[-1][0:3]
            for idx, category in enumerate(MyConfig.category_list):
                if label == category:
                    label = idx

            filename = img_path.split(os.path.sep)[-1]
            return img, label, filename
        except :
            print("오류 난 이미지 : ", img_path.split(os.path.sep)[-1])

    def __len__(self):
        return len(self.data_list)


def train_model(device, model, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs = MyConfig.n_epochs):
    with open(MyConfig.LOGGING_FILE, 'a+t') as log:
        print("Training start..")
        log.write("Training start..\n")
        start = time.time()
        criterion = MyConfig.criterion
        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0
        y_true = []
        y_pred = []
        patience = 10 # 높은 loss 학습 누적 최대치
        early_stopping = EarlyStopping(patience, verbose=True)
        
        # ckpt 변수 
        loss_names = []
        all_losses = []
        all_acc = []
        all_epochs = []
        
        for epoch in range(num_epochs):
            print('-' * 100)
            print("Epoch {}/{}".format(epoch, num_epochs-1))
            log.write('-' * 100)
            log.write("\nEpoch {}/{}".format(epoch, num_epochs-1))

            ep_start = time.time() 
            epoch_loss = 0.0
            # Each epoch has a training and valid phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                train_loss = 0.0
                correct_pred = 0
                # train the model
                for image, label, filename in dataloaders[phase]:
                    image = image.to(device)
                    label = label.to(device)

                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(image)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, label)
                        # backward + optimize only if in training phase
                        y_true.append(label.detach().cpu().numpy())
                        y_pred.append(preds.detach().cpu().numpy())
                        m = MultiLabelBinarizer().fit(y_true) 
                        
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
            
                    # statistics
                    train_loss += loss.item() * image.size(0)
                    correct_pred += torch.sum(preds == label.data)
                if phase == 'train':
                    scheduler.step()

                epoch_loss = train_loss / dataset_sizes[phase]
                epoch_acc = correct_pred.double() / dataset_sizes[phase]
                
                loss_names.append(phase)
                # all_losses.append(torch.FloatTensor([epoch_loss]))   
                # all_acc.append(torch.FloatTensor([epoch_acc]))  
                all_losses.append(epoch_loss)  
                all_acc.append(epoch_acc.cpu().numpy())

                print('{} Loss: {:.4f} Acc: {:.2f}%'.format(phase, epoch_loss, epoch_acc*100))
                log.write('\n{} Loss: {:.4f} Acc: {:.2f}%'.format(phase, epoch_loss, epoch_acc*100))
                
                if phase == 'val' :
                    pre = precision_score(m.transform(y_true), m.transform(y_pred), average='micro')
                    recall = recall_score(m.transform(y_true), m.transform(y_pred), average='micro')
                    f1 = f1_score(m.transform(y_true), m.transform(y_pred), average='micro')
                    
                    log.write('\n_precision= {}\n'.format(round(pre,4)))
                    log.write('_recall= {}\n'.format(round(recall,4)))
                    log.write('_f1= {}\n'.format(round(f1,4)))
                    # deep copy the model        
                    print('_precision= {}'.format(round(pre,4)))
                    print('_recall= {}'.format(round(recall,4)))
                    print('_f1= {}'.format(round(f1,4)))
                
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_epoch = epoch
                    best_model_wts = copy.deepcopy(model.state_dict())

            all_epochs.append(epoch)

            early_stopping(epoch_loss, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            # model save 
            if epoch % 1 == 0 :
                MODEL_PATH = os.path.join(MyConfig.MODEL_PATH, 'NIA_ship_DenseNet_{}.pth'.format(epoch))
                torch.save(model.state_dict(), MODEL_PATH)

            ep_end = time.time()
            print('Time : {}'.format(datetime.timedelta(seconds=ep_end-ep_start)))
            log.write('\nTime : {}\n'.format(datetime.timedelta(seconds=ep_end-ep_start)))

        BEST_MODEL_PATH = os.path.join(MyConfig.MODEL_PATH, 'NIA_ship_model_Best_{}.pth'.format(best_epoch))
        torch.save(best_model_wts, BEST_MODEL_PATH)
        
        ckpt_PATH = os.path.join(MyConfig.MODEL_PATH, 'NIA_ship_DenseNet_ckpt.pt')
        
        
        print("----------------ckpt 저장-----------------")
        
        torch.save({'epoch' : all_epochs,
            'loss' : all_losses,
            'acc':all_acc},
            ckpt_PATH
        )
        end = time.time()
        log.write('=' * 100)
        log.write('\n[Training Complete: {}]'.format(datetime.timedelta(seconds=end-start)))
        print('=' * 100)
        print('[Training Complete: {}]'.format(datetime.timedelta(seconds=end-start)))
        

def predict_model(pre_model,dataloaders) :
    y_pred = []
    y_true = []
    file_name = []
    pre_model.eval()
    start = time.time()
    print("추론중... ")
    # iterate over test data
    for image, labels , filename in dataloaders['test']:
        output = pre_model(image) # Feed Network
        output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
        y_pred.extend(output) # Save Prediction
        labels = labels.data.cpu().numpy()
        y_true.extend(labels) # Save Truth
        file_name.extend(filename)
     
    # constant for classes
    classes = ('101','201','202','203','204','205','206','207','301','302','303')

    # Build confusion matrix 
    # d = {'file':file_name ,'y_true':y_true ,'y_pred':y_pred}
    # df_match = pd.DataFrame(d) ### excel 형식으로 만들기 
    # df_match.to_csv("./match_table.csv", sep=",")
    end = time.time()
    print('[predict Complete: {}]'.format(datetime.timedelta(seconds=end-start)))
    # for true,pred,file in zip(y_true,y_pred,file_name):
    #     if true != pred :
    #         print("---------------------------")
    #         print("오탐지 파일명 : ", file)
    #         print("\n정답값: ", classes[true] , ", 잘못추론한 값 : ", classes[pred])
    #         print("---------------------------")
    #         shutil.copy(os.path.join(os.getcwd(),"datasets","test",file),os.path.join(os.getcwd(),"datasets","error",file))
    
    cf_matrix = confusion_matrix(y_true, y_pred)
    # df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) *10, index = [i for i in classes],
    #                     columns = [i for i in classes]) # 비율
    df_cm = pd.DataFrame(cf_matrix, index = [i for i in classes],
                        columns = [i for i in classes]) # 수치 
    plt.figure(figsize = (12,7))
    sn.heatmap(df_cm, annot=True)
    plt.title('ship predict matrix', fontsize=15)
    plt.savefig('output.png')
    # plt.show()

def ckpt_plot(checkpoint_epoch,checkpoint_loss,checkpoint_acc):

    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []
    
    print("checkpoint_epoch >>> ",checkpoint_epoch) # 최종 epochs
 
    for index in range(len(checkpoint_loss)) :
        if index % 2 == 0 :
            train_loss.append(checkpoint_loss[index])
        else :    
            val_loss.append(checkpoint_loss[index])

    for index in range(len(checkpoint_acc)) :
        if index % 2 == 0 :
            train_acc.append(checkpoint_acc[index])
        else :    
            val_acc.append(checkpoint_acc[index])
            
    # train_loss, val_loss
    plt.figure(figsize=(10,5))
    plt.plot(train_loss,'r', label="train_loss")
    plt.plot(val_loss,'g', label="val_loss")
    plt.legend(ncol=2, loc="upper right")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig('loss.png')
    
    # train_acc, val_acc
    plt.figure(figsize=(10,5))
    plt.plot(train_acc, label="train_acc")
    plt.plot(val_acc, label="val_acc")
    plt.legend(ncol=2, loc="upper right")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()
    plt.savefig('acc.png')
    
               
def main():
    train_list = making_data_list(MyConfig.train_dir)
    valid_list = making_data_list(MyConfig.valid_dir)
    test_list = making_data_list(MyConfig.test_dir)
    print(f"Train Data : {len(train_list)} Images\nValid Data : {len(valid_list)} Images\nTest Data : {len(test_list)} Images\n")

    train_data = MyDataset(train_list, transform=MyConfig.my_transforms)
    valid_data = MyDataset(valid_list, transform=MyConfig.my_transforms)
    test_data = MyDataset(test_list, transform=MyConfig.my_transforms)
    image_datasets = {'train' : train_data, 'val' : valid_data, 'test' : test_data}

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=MyConfig.batch_size, shuffle=True,
                                                 num_workers=MyConfig.num_workers) for x in ['train', 'val' , 'test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val' , 'test']}
    class_num = len(MyConfig.category_list)
    
    
    Task = int(input("작업할 Task 숫자입력 ->(1. Train , 2. Predict , 3. Acc/Loss  : " ))
    
    if Task == 1 :
        make_dirs() 
        # device = 'cuda' if torch.cuda.is_available() else 'cpu'
        device = torch.device(f'cuda:{MyConfig.GPU_NUM}')
        model = densenet121(pretrained=True)
        set_parameter_requires_grad(model, feature_extracting=False)
        model.classifier = nn.Linear(1024, class_num)
        model.to(device)

        optimizer = optim.Adam(model.parameters(), lr=MyConfig.lr)  
        lr_sche = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: 0.95 ** epoch, last_epoch=-1, verbose=False)

        train_model(device, model, optimizer, lr_sche, dataloaders, dataset_sizes) # 학습함수 (추론시 막기)
    
    elif Task == 2 : 
        # 추론 모델 불러오기# dense net -> no fc 
        pre_model = densenet121(pretrained=True)
        pre_model.classifier = nn.Linear(1024, class_num)
        pre_model.load_state_dict(torch.load('./NIA_ship_DenseNet_5.pth'), strict=False) # 사전 학습된 모델을 추론을 위하여 가져옴 
        
        predict_model(pre_model, dataloaders) # 추론함수

    elif Task == 3 :
        # 체크포인트 모델 불러오기
        checkpoint = torch.load('./NIA_ship_DenseNet_ckpt.pt')
        checkpoint_epoch = checkpoint["epoch"]
        checkpoint_loss = checkpoint["loss"]
        checkpoint_acc = checkpoint["acc"]

        ckpt_plot(checkpoint_epoch,checkpoint_loss,checkpoint_acc) # 그래프함수
      


if __name__=='__main__':
    main()
