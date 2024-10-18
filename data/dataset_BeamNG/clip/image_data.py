import os
import glob

path="E:/Python_Pro/clip/dataset/vision"
def image_rename(folder_path):
    files = os.listdir(folder_path)
    for filename in files:
        if filename.endswith(".PNG") and filename.startswith("M001__"):
            timestamp=filename.split("__")[4].split("_")[0]
            count=0
            new_name = timestamp+"_"+str(count)+".PNG"
            new_name_exist=os.path.join(path,new_name)
            while os.path.exists(new_name_exist):
                count+=1
                new_name=timestamp+"_"+str(count)+".PNG"
                new_name_exist = os.path.join(folder_path, new_name)
            old_path = os.path.join(folder_path, filename)
            new_path = os.path.join(folder_path, new_name)
            os.rename(old_path, new_path)

def time_calc(time):
    time = time[8:]
    time=time.split("_")[0]
    time_num=int(time[0:2])*3600+int(time[2:4])*60+int(time[4:6])
    return int(time_num)


if __name__=="__main__":
    start=time_calc("20240719165752_0.PNG")
    end=time_calc("20240719165745")
    print(start-end)
