import re
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from collections import OrderedDict
from utils import load_state_dict_from_url
from torch import Tensor
from typing import Any, List, Tuple
#####################################################
import gc
import os
import time
import datetime
from tkinter import X
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
# from torchvision.models import densenet121
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
    data_root = os.path.join(os.getcwd(),'dataset','final')
    dataset_dir ='dataset'
    train_dir = 'train'
    valid_dir = 'valid'
    test_dir = 'test'
    category_list = ['098L','098R','099L','099R','100R','100L']
    now = datetime.datetime.now().strftime('%y%m%d')
    time = datetime.datetime.now().strftime('%H%M')
    MODEL_PATH = os.path.join(os.getcwd(), 'model', 'morals_DenseNet', now+'_'+time)
    txt_log_dir = os.path.join(os.getcwd(), 'log')
    LOGGING_FILE = os.path.join(txt_log_dir, now+'_'+time+'_morals_DenseNet_training_log.txt')
    GPU_NUM = 0
    IMG_SIZE = (224, 224)

    my_transforms = transforms.Compose([           
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        # transforms.Normalize(IMG_MEAN, IMG_STD)
    ])
    # transforms_no_resize = transforms.Compose([                  
    #     transforms.ToTensor(),
    #     # transforms.Normalize(IMG_MEAN, IMG_STD)
    # ])
        
    criterion = nn.CrossEntropyLoss()
    num_workers = 0 # cpu서 학습시엔 0으로 
    batch_size = 16
    lr = 1e-4
    n_epochs = 100
    ct = 100


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


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
  
def making_data_list(dataset_dir, dataset):
    image_root = os.path.join(MyConfig.data_root, dataset_dir, dataset)
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
        img_path = self.data_list[index]  
        img = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        label = img_path.split(os.path.sep)[-1][0:4]
        for idx, category in enumerate(MyConfig.category_list):
            if label == category:
                label = idx

        filename = img_path.split(os.path.sep)[-1]
        return img, label, filename

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
        patience = MyConfig.ct # 높은 loss 누적 최대치
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
                print("Early stopping!")
                break

            # model save 
            if epoch % 1 == 0 :
                MODEL_PATH = os.path.join(MyConfig.MODEL_PATH, 'DenseNet_model_{}.pth'.format(epoch))
                torch.save(model.state_dict(), MODEL_PATH)

            ep_end = time.time()
            print('Time : {}'.format(datetime.timedelta(seconds=ep_end-ep_start)))
            log.write('\nTime : {}\n'.format(datetime.timedelta(seconds=ep_end-ep_start)))

        BEST_MODEL_PATH = os.path.join(MyConfig.MODEL_PATH, 'Best_model_DenseNet_{}.pth'.format(best_epoch))
        torch.save(best_model_wts, BEST_MODEL_PATH)
        
        ckpt_PATH = os.path.join(MyConfig.MODEL_PATH, 'DenseNet_ckpt.pt')
        
        
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
        
#############################################################################################
__all__ = ['DenseNet', 'densenet121']

model_urls = {
    'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth'
}


class _DenseLayer(nn.Module):
    def __init__(
        self,
        num_input_features: int,
        growth_rate: int,
        bn_size: int,
        drop_rate: float,
        memory_efficient: bool = False
    ) -> None:
        super(_DenseLayer, self).__init__()
        self.norm1: nn.BatchNorm2d
        self.add_module('norm1', nn.BatchNorm2d(num_input_features))
        self.relu1: nn.ReLU
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.conv1: nn.Conv2d
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1,
                                           bias=False))
        self.norm2: nn.BatchNorm2d
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate))
        self.relu2: nn.ReLU
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.conv2: nn.Conv2d
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1,
                                           bias=False))
        self.drop_rate = float(drop_rate)
        self.memory_efficient = memory_efficient

    def bn_function(self, inputs: List[Tensor]) -> Tensor:
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        return bottleneck_output

    # todo: rewrite when torchscript supports any
    def any_requires_grad(self, input: List[Tensor]) -> bool:
        for tensor in input:
            if tensor.requires_grad:
                return True
        return False

    @torch.jit.unused  # noqa: T484
    def call_checkpoint_bottleneck(self, input: List[Tensor]) -> Tensor:
        def closure(*inputs):
            return self.bn_function(inputs)

        return cp.checkpoint(closure, *input)

    @torch.jit._overload_method  # noqa: F811
    def forward(self, input: List[Tensor]) -> Tensor:
        pass

    @torch.jit._overload_method  # noqa: F811
    def forward(self, input: Tensor) -> Tensor:
        pass

    # torchscript does not yet support *args, so we overload method
    # allowing it to take either a List[Tensor] or single Tensor
    def forward(self, input: Tensor) -> Tensor:  # noqa: F811
        if isinstance(input, Tensor):
            prev_features = [input]
        else:
            prev_features = input

        if self.memory_efficient and self.any_requires_grad(prev_features):
            if torch.jit.is_scripting():
                raise Exception("Memory Efficient not supported in JIT")

            bottleneck_output = self.call_checkpoint_bottleneck(prev_features)
        else:
            bottleneck_output = self.bn_function(prev_features)

        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                                     training=self.training)
        return new_features


class _DenseBlock(nn.ModuleDict):
    _version = 2

    def __init__(
        self,
        num_layers: int,
        num_input_features: int,
        bn_size: int,
        growth_rate: int,
        drop_rate: float,
        memory_efficient: bool = False
    ) -> None:
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features: Tensor) -> Tensor:
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features: int, num_output_features: int) -> None:
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    def __init__(
        self,
        growth_rate: int = 32,
        block_config: Tuple[int, int, int, int] = (6, 12, 24, 16),
        num_init_features: int = 64,
        bn_size: int = 4,
        drop_rate: float = 0.1,
        num_classes: int = 1000,
        memory_efficient: bool = False
    ) -> None:

        super(DenseNet, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2,
                                padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out


def _load_state_dict(model: nn.Module, model_url: str, progress: bool) -> None:
    # '.'s are no longer allowed in module names, but previous _DenseLayer
    # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
    # They are also in the checkpoints in model_urls. This pattern is used
    # to find such keys.
    pattern = re.compile(
        r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')

    state_dict = load_state_dict_from_url(model_url, progress=progress)
    for key in list(state_dict.keys()):
        res = pattern.match(key)
        if res:
            new_key = res.group(1) + res.group(2)
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
    model.load_state_dict(state_dict)


def _densenet(
    arch: str,
    growth_rate: int,
    block_config: Tuple[int, int, int, int],
    num_init_features: int,
    pretrained: bool,
    progress: bool,
    **kwargs: Any
) -> DenseNet:
    model = DenseNet(growth_rate, block_config, num_init_features, **kwargs)
    if pretrained:
        _load_state_dict(model, model_urls[arch], progress)
    return model


def densenet121(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> DenseNet:
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_.
    """
    return _densenet('densenet121', 32, (6, 12, 24, 16), 64, pretrained, progress,
                     **kwargs)

################################################################################################
def predict_model(pre_model,dataloaders) :
    y_pred = []
    y_true = []
    file_name = []
    pre_model.eval()
    # iterate over test data
    for image, labels , filename in dataloaders['test']:
        output = pre_model(image) # Feed Network
        output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
        y_pred.extend(output) # Save Prediction
        labels = labels.data.cpu().numpy()
        y_true.extend(labels) # Save Truth
        file_name.extend(filename)
      
    # print("정답 : ", y_true) 
    # print("추론된 라벨 : ", y_pred)
    # print("파일명 : " , file_name)
    # constant for classes
    classes = ('098L','098R','099L','099R','100R','100L')

    # Build confusion matrix 
    
    cf_matrix = confusion_matrix(y_true, y_pred)
    # df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) *10, index = [i for i in classes],
    #                     columns = [i for i in classes]) # 비율
    df_cm = pd.DataFrame(cf_matrix, index = [i for i in classes],
                        columns = [i for i in classes]) # 수치 
    plt.figure(figsize = (12,7))
    sn.heatmap(df_cm, annot=True)
    plt.title('morals predict matrix', fontsize=15)
    plt.savefig('output.png')
    plt.show()

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
    plt.show()
    
    # train_acc, val_acc
    plt.figure(figsize=(10,5))
    plt.plot(train_acc, label="train_acc")
    plt.plot(val_acc, label="val_acc")
    plt.legend(ncol=2, loc="upper right")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()
    plt.savefig('acc.png')
    plt.show()
    
        
            
def main():
    train_list = making_data_list(MyConfig.dataset_dir, MyConfig.train_dir)
    valid_list = making_data_list(MyConfig.dataset_dir, MyConfig.valid_dir)
    test_list = making_data_list(MyConfig.dataset_dir, MyConfig.test_dir)
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
        device = torch.device(f'cuda:{MyConfig.GPU_NUM}')
        model = densenet121(pretrained=True)
        set_parameter_requires_grad(model, feature_extracting=False)
        model.classifier = nn.Linear(1024, class_num)
        model.to(device)

        optimizer = optim.Adam(model.parameters(), lr=MyConfig.lr)  
        lr_sche = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: 0.95 ** epoch, last_epoch=-1, verbose=False)
        train_model(device, model, optimizer, lr_sche, dataloaders, dataset_sizes) # 학습함수
        
    elif Task == 2 :  
        ######## 메모리 최적화를 위한 코드##########################
        gc.collect()
        torch.cuda.empty_cache()
        ##########################################################
        
        # 추론 모델 불러오기# dense net -> no fc 
        pre_model = densenet121(pretrained=True)
        pre_model.classifier = nn.Linear(1024, class_num)
        
        pre_model.load_state_dict(torch.load('최종학습_no_crop/Best_model_DenseNet_35.pth'), strict=False) # 사전 학습된 모델을 추론을 위하여 가져옴 
        
        predict_model(pre_model, dataloaders) # 추론함수


    elif Task == 3 :
        #체크포인트 모델 불러오기
        
        checkpoint = torch.load('최종학습_no_crop/DenseNet_ckpt.pt')
        checkpoint_epoch = checkpoint["epoch"]
        checkpoint_loss = checkpoint["loss"]
        checkpoint_acc = checkpoint["acc"]

        ckpt_plot(checkpoint_epoch,checkpoint_loss,checkpoint_acc) # 그래프함수


if __name__=='__main__':
    main()
