import random
import numpy as np
import os 
from PIL import Image 
from PIL import ImageFilter

# 새로 만들 이미지의 갯수 정해주기 
num_augmented_images = 25
# 현재 파일 폴더 경로
current_path = os.getcwd()
# 원본 사진이 존재하는 폴더 경로 
file_path = current_path + '\\save_file\\99\\' 

filter_list = [ImageFilter.BLUR, ImageFilter.DETAIL, 
ImageFilter.EDGE_ENHANCE,
ImageFilter.SHARPEN,ImageFilter.SMOOTH]

# 위의 폴더 내부에 있는 이미지 이름들을 list로 저장 
file_names = os.listdir(file_path)
# 폴더 내부의 원본 이미지의 길이 저장 
total_origin_image_num = len(file_names)  #5

# augmentation count 횟수
augment_cnt = 1

for i in range(0,num_augmented_images) :
# 폴더 내의 전체 원본 이미지 개수 중 하나를 랜덤하게 선택하여 file_name 에 인덱스를 갖는 이미지 이름 저장 
    # 변환시킬 이미지를 랜덤하게 index 선택하기  
    change_picture_index = random.randrange(0,total_origin_image_num) 
    # 랜덤하게 선택된 index 의 파일명 
    file_name = file_names[change_picture_index]


    # 원본 데이터 파일 경로 
    origin_image_path = 'save_file\\99\\' + file_name

    # image open 하기
    image = Image.open(origin_image_path)
    # 이미지 회전 각도 -10 ~ 10 도 사이에서 random 하게 설정 
    rotated_image = image.rotate(random.randrange(-10,10))
    # filter_img = rotated_image.filter(filter_list[i])
    
    # augmentation 이미지를 원본 파일이 존재하는 위치와 동일한 위치에 저장 
    rotated_image.save(file_path + file_name + '_' + str(augment_cnt) + '.png')

    augment_cnt += 1
    
