import os


def img_remove(test_dir,valid_dir,train_dir):
	train_list = os.listdir(train_dir)
	valid_list = os.listdir(valid_dir)
	test_list = os.listdir(test_dir)
	with open("error.txt", 'r') as file:
		remove_file_list= file.read().splitlines()

		for i in remove_file_list :
			for list in test_list :		
				if list == i : 
					path = os.path.join(test_dir,list)
					os.remove(path)
					print(list," 삭제완료.")




if __name__ == "__main__":
	test_dir = os.path.join(os.getcwd(),"dataset","one_cycle2","dataset","test")
	valid_dir = os.path.join(os.getcwd(),"dataset","one_cycle2","dataset","valid")
	train_dir = os.path.join(os.getcwd(),"dataset","one_cycle2","dataset","train")
 
 
	img_remove(test_dir,valid_dir,train_dir)