import os
import json
import shutil
from PIL import Image
from sklearn.model_selection import train_test_split

# 전체 데이터셋 중 대구치만 추출하는 함수
def morals_choice(json_path,images_path,save_path) :
	add_list_98_R = []
	add_list_99_R = []
	add_list_100_R = []
	add_list_98_L = []
	add_list_99_L = []
	add_list_100_L = []
 
	with open(json_path, 'r') as file:
		data = json.load(file)
		anno_list = data["annotations"]
		# 대구치 사진만 98 ~ 100 별 추출
		for i in anno_list :
			if (i["category_id"] == 98) :
				if ("right" in i["image_id"]):
					add_list_98_R.append(i["image_id"])
				if ("left" in i["image_id"]):
					add_list_98_L.append(i["image_id"])
			if (i["category_id"] == 99) :
				if ("right" in i["image_id"]):
					add_list_99_R.append(i["image_id"])
				if ("left" in i["image_id"]):
					add_list_99_L.append(i["image_id"])
      
			if (i["category_id"] == 100) :
				if ("right" in i["image_id"]):
					add_list_100_R.append(i["image_id"])
				if ("left" in i["image_id"]):
					add_list_100_L.append(i["image_id"])
      
      
		result1 = dict.fromkeys(add_list_98_R)
		file_list_98_R = list(result1)
		print("98번 오른쪽 카테고리 수 : ",len(file_list_98_R))	
		result2 = dict.fromkeys(add_list_99_R)
		file_list_99_R = list(result2)
		print("99번 오른쪽 카테고리 수 : ",len(file_list_99_R))
		result3 = dict.fromkeys(add_list_100_R)
		file_list_100_R = list(result3)
		print("100번 오른쪽 카테고리 수 : ",len(file_list_100_R))
  
		result4 = dict.fromkeys(add_list_98_L)
		file_list_98_L = list(result4)
		print("98번 왼쪽 카테고리 수 : ",len(file_list_98_L))	
		result5 = dict.fromkeys(add_list_99_L)
		file_list_99_L = list(result5)
		print("99번 왼쪽 카테고리 수 : ",len(file_list_99_L))
		result6 = dict.fromkeys(add_list_100_L)
		file_list_100_L = list(result6)
		print("100번 왼쪽 카테고리 수 : ",len(file_list_100_L))
  
  
		files = os.listdir(images_path)
  
  
		for file in files:
			if file in file_list_98_R:
				shutil.copy(images_path +'\\'+ file, save_path +'\\'+'R'+'\\'+'98'+'\\098R_'+ file)
				# print('{} has been copied in new folder!'.format(file))
			if file in file_list_99_R:
				shutil.copy(images_path +'\\'+ file, save_path +'\\'+'R'+'\\'+'99'+'\\099R_'+ file)
				# print('{} has been copied in new folder!'.format(file))
			if file in file_list_100_R:
				shutil.copy(images_path +'\\'+ file, save_path +'\\'+'R'+'\\'+'100'+'\\100R_'+ file)
				# print('{} has been copied in new folder!'.format(file))
			if file in file_list_98_L:
				shutil.copy(images_path +'\\'+ file, save_path +'\\'+'L'+'\\'+'98'+'\\098L_'+ file)
				# print('{} has been copied in new folder!'.format(file))
			if file in file_list_99_L:
				shutil.copy(images_path +'\\'+ file, save_path +'\\'+'L'+'\\'+'99'+'\\099L_'+ file)
				# print('{} has been copied in new folder!'.format(file))
			if file in file_list_100_L:
				shutil.copy(images_path +'\\'+ file, save_path +'\\'+'L'+'\\'+'100'+'\\100L_'+ file)
				# print('{} has been copied in new folder!'.format(file))
			
	print("end!!")

# 뽑힌 대구치 이미지를 crop하는 함수 
def image_crop(save_path,r_crop_path,l_crop_path):
	categorys = ['98','99','100']
	for category in categorys :
		L_path = os.path.join(save_path,'L',category)
		R_path = os.path.join(save_path,'R',category)
		L_images = os.listdir(L_path)
		R_images = os.listdir(R_path)


		for image in L_images:
			# print(category+"L 시작")
			img = Image.open(L_path+"\\"+image)
			# print(img.size)
	
			frac_top_bottom = 0.76 # 위 - 아래 사이즈 조절 
			left = 0
			right = img.size[0]*0.5
			upper = img.size[1]*((1-frac_top_bottom)/2)
			bottom = img.size[1]-((1-frac_top_bottom)/2)*img.size[1]

			# crop을 통해 이미지 자르기       (left,up, rigth, down)
			cropped_img = img.crop((left, upper, right, bottom))

			# croppedImage.show()
			# print("잘려진 사진 크기 :", cropped_img.size)
			cropped_img.save(l_crop_path+"\\"+category+"\\"+image)

		for image2 in R_images:
			# print(category+"R 시작")
			img2 = Image.open(R_path+"\\"+image2)
			# print(img2.size)

			frac_top_bottom = 0.76 # 위 - 아래 사이즈 조절 
			left = img2.size[0]*0.35
			right = img2.size[0]
			upper = img2.size[1]*((1-frac_top_bottom)/2)
			bottom = img2.size[1]-((1-frac_top_bottom)/2)*img2.size[1]

			# crop을 통해 이미지 자르기       (left,up, rigth, down)
			cropped_img = img2.crop((left, upper, right, bottom))

			# croppedImage.show()
			# print("잘려진 사진 크기 :", cropped_img.size)
			cropped_img.save(r_crop_path+"\\"+category+"\\"+image2)
	print('end!!')
 
 
def _copyfileobj_patched(fsrc, fdst, length=16*1024*1024):
	while 1:
		buf = fsrc.read(length)
		if not buf:
			break
		fdst.write(buf)
# 분류 학습을 위해 데이터셋화 하는 함수 
def dataset_made(r_crop_path,l_crop_path,train_dir,valid_dir,test_dir) : 
	categorys = ['98','99','100']
	for category in categorys :
		L_path = os.path.join(l_crop_path,category)
		R_path = os.path.join(r_crop_path,category)
  
		if not os.path.exists(L_path):
			os.makedirs(L_path)
		if not os.path.exists(R_path):
			os.makedirs(R_path)
   
		L_data = os.listdir(L_path)
		R_data = os.listdir(R_path)
  
  
		L_dataset_train_list, L_dataset_another_list = train_test_split(L_data, test_size=0.2, random_state=0)
		L_dataset_test_list, L_dataset_valid_list = train_test_split(L_dataset_another_list, test_size=0.5, random_state=0)
  
		R_dataset_train_list, R_dataset_another_list = train_test_split(R_data, test_size=0.2, random_state=0)
		R_dataset_test_list, R_dataset_valid_list = train_test_split(R_dataset_another_list, test_size=0.5, random_state=0)
		shutil.copyfileobj = _copyfileobj_patched
		for file in R_dataset_train_list:
				shutil.move(r_crop_path +'\\'+category+"\\"+ file, train_dir+"\\"+file)
		for file1 in R_dataset_test_list:
				shutil.move(r_crop_path +'\\'+category+"\\"+ file1, test_dir+"\\"+file1)
		for file2 in R_dataset_valid_list:
				shutil.move(r_crop_path +'\\'+category+"\\"+ file2, valid_dir+"\\"+file2)
    
		for file3 in L_dataset_train_list:
				shutil.move(l_crop_path +'\\'+category+"\\"+ file3, train_dir+"\\"+ file3)
		for file4 in L_dataset_test_list:
				shutil.move(l_crop_path +'\\'+category+"\\"+ file4, test_dir+"\\"+ file4)
		for file5 in L_dataset_valid_list:
				shutil.move(l_crop_path +'\\'+category+"\\"+ file5, valid_dir+"\\"+ file5)


def createDirectory(save_path,r_crop_path,l_crop_path,dataset_path,train_dir,valid_dir,test_dir):
	try:
		if not os.path.exists(save_path):
			os.makedirs(save_path)

			R_98_path = os.path.join(save_path,'R','98')
			R_99_path = os.path.join(save_path,'R','99')
			R_100_path = os.path.join(save_path,'R','100')
			L_98_path = os.path.join(save_path,'L','98')
			L_99_path = os.path.join(save_path,'L','99')
			L_100_path = os.path.join(save_path,'L','100')
			os.makedirs(R_98_path)
			os.makedirs(R_99_path)
			os.makedirs(R_100_path)
   
			os.makedirs(L_98_path)
			os.makedirs(L_99_path)
			os.makedirs(L_100_path)
   
		if not os.path.exists(r_crop_path):
			os.makedirs(r_crop_path)
		if not os.path.exists(l_crop_path):
			os.makedirs(l_crop_path)
		if not os.path.exists(dataset_path):
			os.makedirs(dataset_path)
		if not os.path.exists(train_dir):
			os.makedirs(train_dir)
		if not os.path.exists(valid_dir):
			os.makedirs(valid_dir)
		if not os.path.exists(test_dir):
			os.makedirs(test_dir)
	except OSError:
		print("Error: Failed to create the directory.")
  
    
if __name__ == "__main__" :
    # 경로지정
	json_path = os.path.join(os.getcwd(),"data.json")
	images_path = "\\\\192.168.219.150/XaiData/R&D/Project/2022_03_스마트_심미_보철/03.데이터/20221128_final_data/images"
    # 경로 생성 
	save_path = os.path.join(os.getcwd(),"dataset","final","morals_image")

	r_crop_path = os.path.join(os.getcwd(),"dataset","final","morals_image",'R')
	l_crop_path = os.path.join(os.getcwd(),"dataset","final","morals_image",'L')
 
	dataset_path = os.path.join(os.getcwd(),"dataset","final","dataset")
	train_dir = os.path.join(dataset_path,'train')
	valid_dir = os.path.join(dataset_path,'valid')
	test_dir = os.path.join(dataset_path,'test')
 
	
	createDirectory(save_path,r_crop_path,l_crop_path,dataset_path,train_dir,valid_dir,test_dir)
	morals_choice(json_path, images_path, save_path) 
	image_crop(save_path,r_crop_path,l_crop_path)
	dataset_made(r_crop_path,l_crop_path,train_dir,valid_dir,test_dir)