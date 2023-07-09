import gc
import os, sys
import datetime
import torch
import torch.nn as nn
from torchvision.models import densenet121
import torchvision.transforms as transforms
# import EarlyStopping
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

class MyConfig:
    IMG_SIZE = (224, 224)

    my_transforms = transforms.Compose([           
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
    ])
            
#### 1 파일을 추론하는 코드 
def one_file_predict(images) :
	print("test 항목 추론 시작 ......... \n")
	pre_model = densenet121(pretrained=True)
	pre_model.classifier = nn.Linear(1024, 6)
	pre_model.load_state_dict(torch.load('Best_model_DenseNet_35.pth'), strict=False) # 사전 학습된 모델을 추론
	pre_model.eval()
	# iterate over test data
	transform = MyConfig.my_transforms
	img = Image.open(images).convert('RGB')
	img = transform(img)
	
	output = pre_model(img.unsqueeze(0)) # Feed Network
	output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
	y_pred = output # Save Prediction
		
	classes = ('1번 클래스 왼쪽','1번 클래스 오른쪽','2번 클래스 왼쪽','2번 클래스 오른쪽','3번 클래스 왼쪽','3번 클래스 오른쪽')  
	print("파일명 >> ",images, " 의 추론된값 = ",classes[y_pred[0]],"\n") 
		
	print("test 항목 추론 완료. ")


           
def main():

	######## 메모리 최적화를 위한 코드##########################
	gc.collect()
	torch.cuda.empty_cache()
	##########################################################

	images = sys.argv[1]
	# images = "sample.png"
	one_file_predict(images)

if __name__=='__main__':
    main()
