import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from collections import Counter

def processing_channel(df, channel, train_size=0.7):
    # Filter and sort by channel and time
    cols=[]
    for col in df.columns:
        if df[col].dtype == "object":
            cols.append(col)
    
    df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')
    df=df.fillna(0)
    df=df[df['channel'] == channel].copy()
    print(f"Total samples {df.shape[0]}")
    
    temp = df[df['channel'] == channel].copy()
    print(f"TEMP SHAPE: {temp.shape[0]}")

    temp = temp.sort_values(by='time')
    # Filter zero values (clean data)
    temp=temp[(temp["Carrier_Doppler_hz"]==0)&(temp["Pseudorange_m"]==0)&(temp["TOW"]==0)&(temp["Carrier_phase"]==0)&(temp["EC"]==0)&(temp["LC"]==0)&(temp["PC"]==0)&(temp["PIP"]==0)&(temp["PQP"]==0)&(temp["TCD"]==0)&(temp["CN0"]==0)]        
    # Remove processed times from original df
    new_df = df[~df.time.isin(temp.time)]
    print(f"NEW_DF SHAPE: {new_df.shape[0]}")
    print(new_df.spoofed.value_counts())

    # Split into train/validation while maintaining temporal order
    new_df = new_df.sort_values(by='time')

    time_1 = new_df[new_df.spoofed == 1].time.values
    time_0 = new_df[new_df.spoofed == 0].time.values
    x=int(train_size * len(time_1))
    train_time_1 = time_1[:x]
    train_time_0 = time_0[:x]
    val_time_1 = time_1[x:]
    val_time_0 = time_0[x:len(time_1)]

    print(len(train_time_1), len(train_time_0), len(val_time_1), len(val_time_0))

    new_df = None
    i=0
    print(f"Creating training set ")
    print(f"  Total unique times for class 1 in training set: {len(train_time_1)}")
    for target_time in train_time_1:
        if new_df is None:
            new_df = df[(df.time <= target_time) & (df.time >= target_time - 100)]
            new_df["batch_id"]=i
        else:
            new_temp=df[(df.time <= target_time) & (df.time >= target_time - 100)]
            new_temp["batch_id"]=i
            new_df = pd.concat([new_df, new_temp], ignore_index=True)
        i+=1
        if(i%100==0):
            print(f"  Processed Training {i} samples")
    for target_time in train_time_0:
        new_temp=df[(df.time <= target_time) & (df.time >= target_time - 100)]
        new_temp["batch_id"]=i
        new_df = pd.concat([new_df, new_temp], ignore_index=True)
        i+=1
        if(i%100==0):
            print(f"  Processed Training {i} samples")

    train_df = new_df.copy()

    new_df = None
    print(f"Creating validation set ")
    print(f"  Total unique times for class 0 in training set: {len(train_time_0)}")

    for target_time in val_time_1:
        if new_df is None:
            new_df = df[(df.time <= target_time) & (df.time >= target_time - 100)]
            new_df["batch_id"]=i
        else:
            new_temp=df[(df.time <= target_time) & (df.time >= target_time - 100)]
            new_temp["batch_id"]=i
            new_df = pd.concat([new_df, new_temp], ignore_index=True)
        i+=1
        if(i%100==0):
            print(f"  Processed Validation {i} samples")

    for target_time in val_time_0:
        new_temp=df[(df.time <= target_time) & (df.time >= target_time - 100)]
        new_temp["batch_id"]=i
        new_df = pd.concat([new_df, new_temp], ignore_index=True)
        i+=1
        if(i%100==0):
            print(f"  Processed Validation {i} samples")
    
    val_df = new_df.copy()
    print(f"Training set class distribution for channel {channel,train_df.shape[0]}:")
    print(train_df.spoofed.value_counts())
    print(f"Validation set class distribution for channel {channel,val_df.shape[0]}:")
    print(val_df.spoofed.value_counts())

    print("="*70)
    
    return train_df, val_df

df=pd.read_csv("./dataset/train.csv")

for i in range(0, 8):
    tr,va=processing_channel(df, channel=f"ch{i}", train_size=0.7)
    tr.to_csv(f"./dataset/train/train_ch{i}.csv", index=False)
    va.to_csv(f"./dataset/val/val_ch{i}.csv", index=False)