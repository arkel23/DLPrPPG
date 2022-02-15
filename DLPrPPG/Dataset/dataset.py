import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import samplerate
import numpy as np
import math
import os
import shutil
import random
from PIL import Image
import glob
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import find_peaks
from scipy import stats as st
import scipy.sparse
import neurokit2 as nk
import matplotlib.pyplot as plt
    
def split_data(path, rppg):
    if not os.path.isdir(path):
        os.makedirs(path)
    os.makedirs(os.path.join(path, 'train', rppg), exist_ok=True)
    os.makedirs(os.path.join(path, 'train', 'gt', rppg), exist_ok=True)
    os.makedirs(os.path.join(path, 'test', rppg), exist_ok=True)
    os.makedirs(os.path.join(path, 'test', 'gt', rppg), exist_ok=True)

    raw_rppg_path = os.path.join('./Data/rppg/All_file', rppg)
    raw_ppg_path = os.path.join('./Data/ppg', rppg)
    #skip = True
    for path_, dirs, files in os.walk(raw_rppg_path):
        # if skip==True:
        #    skip=False
        # else:
        for filename in files:
            id = int(''.join(filter(str.isdigit, filename[:3])))
            print(id)
            if id > 37:
                # print(os.path.join(path,'train',rppg,filename))
                shutil.copyfile(os.path.join(raw_rppg_path, filename),
                                os.path.join(path, 'test', rppg, filename))
                shutil.copyfile(os.path.join(raw_ppg_path, filename), os.path.join(
                    path, 'test', 'gt', rppg, filename))
            else:
                shutil.copyfile(os.path.join(raw_rppg_path, filename), os.path.join(
                    path, 'train', rppg, filename))
                shutil.copyfile(os.path.join(raw_ppg_path, filename), os.path.join(
                    path, 'train', 'gt', rppg, filename))
        # print(os.path.join(path,'train','gt',rppg,filename))


def MaxMinNormalization(x):
    x1 = x.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1)).fit(x1)
    out = scaler.transform(x1).reshape(-1)
    #x = (x - Min) / (Max - Min)
    return out

def peak_detect(signal, sample_rate):
        # fre = 1/sample_rate
        #peaks, _ = find_peaks(signal, distance=12, height=0.4)  # remove bpm >150    
        #ppg_clean = nk.ppg_clean(signal,sampling_rate=30)
        info = nk.ppg_findpeaks(signal,sampling_rate=30)
        peaks = info["PPG_Peaks"]
        
        return peaks
        
        


def detrend(X, detLambda=10):
        # Smoothness prior approach as in the paper appendix:
        # "An advanced detrending method with application to HRV analysis"
        # by Tarvainen, Ranta-aho and Karjaalainen
        t = X.shape[0]
        l = t/detLambda #lambda
        I = np.identity(t)
        D2 = scipy.sparse.diags([1, -2, 1], [0,1,2],shape=(t-2,t)).toarray() # this works better than spdiags in python
        detrendedX = (I-np.linalg.inv(I+l**2*(np.transpose(D2).dot(D2)))).dot(X)
        return detrendedX

class TrainDataset(Dataset):

    def __init__(self, path, rppg, mode, data_len):
        #self.transform = transform

        self.files_path_noise = os.path.join(path, mode, rppg)
        # print(self.files_path_noise)
        self.files_path_gt = os.path.join(path, mode, 'gt', rppg)
        self.files_n = sorted(
            glob.glob(os.path.join(self.files_path_noise, '*.txt')))
        # print(self.files_n)
        self.files_g = sorted(
            glob.glob(os.path.join(self.files_path_gt, '*.txt')))
        #self.labels = np.loadtxt(txt_file, dtype=int, usecols=1)
        self.data_len = data_len
        self.sigma = 1.
        self.sample_rate = 30.0

    def expand_peaks(self, peaks):
        
        normal_peaks = np.zeros(self.data_len)
        onehot_peaks = np.zeros(self.data_len)
        for peak in peaks:
            nor_peaks = st.norm.pdf(
                np.arange(0, self.data_len), loc=peak, scale=self.sigma)
            normal_peaks = normal_peaks + nor_peaks
            # # onthot expands (-50 ~ +70)
            # st_ix = np.clip(
                # rpeak-50//(1000/self.sampling_rate), 0, self.all_ws)
            # ed_ix = np.clip(
                # rpeak+70//(1000/self.sampling_rate), 0, self.all_ws)
            # st_ix = np.long(st_ix)
            # ed_ix = np.long(ed_ix)
            # onehot_rpeaks[st_ix:ed_ix] = 1
        # scale to [0, 1]
        
        #normal_peaks = MaxMinNormalization(normal_peaks)
        
        return normal_peaks
    
    
        
    
    
    def __len__(self):
        return len(self.files_n)

    def __getitem__(self, idx):

        with open(self.files_n[idx], 'r') as rppg_f:
            next(rppg_f)
            time_rppg = []
            wave_rppg = []
            lines = rppg_f.readlines()
            for index in lines:
                time_rppg.append(float(index.split(' ')[0])/1000)
                wave_rppg.append(float(index.split(' ')[1]))

        with open(self.files_g[idx], 'r') as rppg_gt:
            # print(self.files_g[idx])
            next(rppg_gt)
            time_gt = []
            wave_gt = []
            lines = rppg_gt.readlines()
            for index in lines:
                time_gt.append(float(index.split(' ')[0]))
                wave_gt.append(float(index.split(' ')[1]))
        #time_rppg = np.array(time_rppg)
        wave_rppg = np.array(wave_rppg)
        #time_gt = np.array(time_gt)
        wave_gt = np.array(wave_gt)
        # if len(time_rppg) > len()
        
        ###resample to the same data lens###
        if len(wave_rppg) != self.data_len:
            wave_rppg = samplerate.resample(
                wave_rppg, (self.data_len/10)/(len(wave_rppg)/10), 'sinc_best')

            if len(wave_rppg) == (self.data_len-1):
                # print('rppg:',self.files_n[idx].split('_')[2],len(wave_rppg))
                wave_rppg = np.hstack([wave_rppg, wave_rppg[-1]])

        if len(wave_gt) != self.data_len:
            wave_gt = samplerate.resample(
                wave_gt, (self.data_len/10)/(len(wave_gt)/10), 'sinc_best')
            if len(wave_gt) == (self.data_len-1):
                wave_gt = np.hstack([wave_gt, wave_gt[-1]])
        
        ###detrend###
        #wave_gt1 = wave_gt
        wave_gt = detrend(wave_gt)
        
        
        ###peak detection###
        
        
        # print('ppg:',self.files_g[idx].split('_')[1],len(wave_gt))
        # if len(wave_rppg) != 300:
            # print('rppg:',self.files_n[idx].split('_')[2],len(wave_rppg))
        # if len(wave_gt) != 300:
            # print('ppg:',self.files_g[idx].split('_')[1],len(wave_gt))

        wave_rppg = np.array([MaxMinNormalization(wave_rppg)])
        wave_gt = np.array([MaxMinNormalization(wave_gt)])
        #wave_gt1 = np.array([MaxMinNormalization(wave_gt1)]) 
        # print(wave_gt.shape)    
        # print(np.squeeze(wave_gt).shape)
        ppg_peaks = peak_detect(np.squeeze(wave_gt),self.sample_rate)

        exp_ppg_peaks = self.expand_peaks(ppg_peaks)
        
        exp_ppg_peaks = np.array([MaxMinNormalization(exp_ppg_peaks)])
        # x = np.arange(300)
        # plt.plot(x,wave_rppg.squeeze())
        # plt.plot(x,wave_gt.squeeze())
        # plt.show()
        # print(wave_rppg.shape)
    
        return wave_rppg, wave_gt,exp_ppg_peaks#, str(self.files_n[idx].split("/")[-1]), str(self.files_g[idx].split("/")[-1])

