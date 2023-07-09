import gc
import os, sys
import torch
import torch.nn as nn
from torchvision.models import densenet121
import torchvision.transforms as transforms
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

class MyConfig:
    IMG_SIZE = (224, 224) # 학습과 추론 성능향상을 위한 

    my_transforms = transforms.Compose([           
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
    ])
            
#### 1 파일을 추론하는 코드 
def one_file_predict(images) :
	print("test 항목 추론 시작 ......... \n")
	output_json = {}
	pre_model = densenet121(pretrained=True)
	pre_model.classifier = nn.Linear(1024, 6) # 6 클래스 고정 
	pre_model.load_state_dict(torch.load('model/moral_model_DenseNet.pth'), strict=False) 
 	# 대구치 분류로 학습된 모델을 불러와 추론 ( Acc : 80.5 % )
	pre_model.eval()
	# iterate over test data
	transform = MyConfig.my_transforms # resize 
	img = Image.open(images).convert('RGB')
	img = transform(img)
	
	# 1 이미지만을 추론했기에 차원을 1단계 하강하여 output을 뽑음  
	output = pre_model(img.unsqueeze(0)) # Feed Network
	output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
	y_pred = output # Save Prediction
	
	# output으로 나온 index값에 따른 클래스 분류 (순서 고정) L:좌측 , R:우측
	classes = ('Class 1_L','Class 1_R','Class 2_L','Class 2_R','Class 3_R','Class 3_L')  
	print("파일명 >> ", images, " 의 추론된값 = ",classes[y_pred[0]],"\n") 
		
	print("test 항목 추론 완료. ")
	output_json = {images:classes[y_pred[0]]}
	return output_json

           
def main():

	######## 메모리 최적화를 위한 코드##########################
	gc.collect()
	torch.cuda.empty_cache()
	##########################################################

	images = sys.argv[1]
	# 대구치 images = "sample.png" 인자로 받아옴 
	output_json = one_file_predict(images)

if __name__=='__main__':
    main()
