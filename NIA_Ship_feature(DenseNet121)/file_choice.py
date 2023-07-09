import os , shutil
 # 경로지정
pwd = "/home/ubuntu/workspace/nia_paint_data/2022000086_export/도막손상"
images_path = pwd

# 경로 생성 
save_path = "/home/ubuntu/workspace/nia_paint_data/2022000086_export/Classification"

category = os.listdir(images_path)

def _copyfileobj_patched(fsrc, fdst, length=16*1024*1024):
    while 1:
        buf = fsrc.read(length)
        if not buf:
            break
        fdst.write(buf)


for i in category :
    shutil.copyfileobj = _copyfileobj_patched
    print(i,"카테고리 시작.")
    list = os.listdir(os.path.join(images_path,i))
    for file in list :
        if os.path.splitext(file)[1] == '.jpg' or os.path.splitext(file)[1] == '.JPG' :
            name = os.path.splitext(file)[0][0:3]
            shutil.copy(images_path+'/'+i+'/'+file , save_path+"/"+name+'/'+file)
    print("clear ", i )