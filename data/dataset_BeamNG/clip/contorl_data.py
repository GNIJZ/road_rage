import os
import pandas as pd



pathold="E:/Python_Pro/clip/dataset/333/received_data_jin.csv"
pathnew="E:/Python_Pro/clip/dataset/333/data_new.csv"

def data_out(pathold,pathnew):
    processed_data = []
    data_old=pd.read_csv(pathold,usecols=[0],header=None)
    data_old=data_old.iloc[:,0].to_list()
    print(data_old[0])
    #print(data_old.iloc[:,0].to_list())
    for i in range(0,len(data_old)):
        time=str(data_old[i])[-6:]
        if "." in time:
            time=time.split(".")[0]
            time=str(time[:-1]+"0"+time[-1])
        value1=str(data_old[i][:6])
        value2=str(data_old[i][6:12])
        value3=str(data_old[i][12:18])
        processed_data.append({
            'time': time,
            'value1': value1,
            'value2': value2,
            'value3': value3
        })


    df = pd.DataFrame(processed_data)
    df.to_csv(pathnew, index=False)

 # df = pd.concat([df_time, df_value1, df_value2, df_value3], axis=1)



if __name__=="__main__":
    data_out(pathold, pathnew)