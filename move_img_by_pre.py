import os


img_dir = "TrainSet/BUS-UCLM_clean/Malignant"
save_dir = "TrainSet/BUS-UCLM_clean/pre"

txt_path = "output/result.txt"

with open(txt_path,'r') as f:
    lines = f.readlines()
    for line in lines:
        name,cid = line.strip().split(".png ")
        read_path = os.path.join(img_dir,name+".png")
        os.makedirs(os.path.join(save_dir,cid),exist_ok=True)
        save_path = os.path.join(save_dir,cid,name+".png")
        os.rename(read_path, save_path)
