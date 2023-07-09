from __future__ import annotations
import os
import shutil
from sklearn.model_selection import train_test_split


# 분류 학습을 위해 데이터셋화 하는 함수 
def dataset_made(images_path,train_dir,test_dir,valid_dir) : 
	# categorys = ['101','201','207']
	categorys = ['101','201','202','203','204','205','206','207','301','302','303']
	for category in categorys :
		print(category," 카테고리 수행중 ......")
		path = os.path.join(images_path,category) # 각 이미지 폴더 


		data = os.listdir(path)
		dataset_train_list, dataset_another = train_test_split(data, test_size=0.2, random_state=0)
		## 8:2로 먼저 Train , Test 를 나눈 후 - > Test 서 50 : 50 으로 Test / Val 분배	
		dataset_test_list, dataset_val_list = train_test_split(dataset_another, test_size=0.5, random_state=0)

		shutil.copyfileobj = _copyfileobj_patched  # shutil 의 copyfileobj 대신 _copyfileobj_patched 이 호출됨
  
		for file in dataset_train_list:
				shutil.move(path +'/'+ file, train_dir+"/"+file)
		for file in dataset_test_list:
				shutil.move(path +'/'+ file, test_dir+"/"+file)
		for file in dataset_val_list:
				shutil.move(path +'/'+ file, valid_dir+"/"+file)
	
		print(category,"- clear ")
		print(len(dataset_train_list),len(dataset_val_list),len(dataset_test_list))
    
    



def createDirectory(dataset_path,train_dir,test_dir,valid_dir):
	try:
		if not os.path.exists(dataset_path):
			os.makedirs(dataset_path)
		if not os.path.exists(train_dir):
			os.makedirs(train_dir)
		if not os.path.exists(test_dir):
			os.makedirs(test_dir)
		if not os.path.exists(valid_dir):
			os.makedirs(valid_dir)
	except OSError:
		print("Error: Failed to create the directory.")

def _copyfileobj_patched(fsrc, fdst, length=16*1024*1024):
    while 1:
        buf = fsrc.read(length)
        if not buf:
            break
        fdst.write(buf)


          
if __name__ == "__main__" :
    # 경로지정
	images_path = "/home/ubuntu/workspace/nia_paint_data/2022000086_export/Classification"
 
    # 경로 생성 
	dataset_path = "/home/ubuntu/workspace/nia_paint_data/2022000086_export/Classification/datasets"
 
	train_dir = os.path.join(dataset_path,'train')
	test_dir = os.path.join(dataset_path,'test')
	valid_dir = os.path.join(dataset_path,'valid')
	createDirectory(dataset_path,train_dir,test_dir,valid_dir)
 
 
	dataset_made(images_path,train_dir,test_dir,valid_dir)