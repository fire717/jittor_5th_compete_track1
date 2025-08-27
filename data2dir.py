import os
import shutil

def copy_file(srcfile,dstfile):
    if not os.path.isfile(srcfile):
        print("%s not exist!"%(srcfile))
    else:
        fpath,fname=os.path.split(dstfile)    #分离文件名和路径
        if not os.path.exists(fpath):
            os.makedirs(fpath)                #创建路径
        shutil.copyfile(srcfile,dstfile)      #复制文件


img_dir = "TrainSet/images/train"
label_path = "TrainSet/labels/trainval.txt"
save_dir = "TrainSet/images_split_dir"
os.makedirs(save_dir,exist_ok=True)


with open(label_path,'r') as f:
    lines = f.readlines()

    for line in lines:
        name,cid = line.strip().split(" ")

        read_path = os.path.join(img_dir,name)
        save_class_dir = os.path.join(save_dir,cid)
        os.makedirs(save_class_dir,exist_ok=True)

        save_path = os.path.join(save_class_dir,name)
        copy_file(read_path,save_path)
