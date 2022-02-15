# -*- coding: UTF-8 -*-
import argparse
import os
import numpy as np
import math
import torch.optim as optim
import torchvision.transforms as transforms
# from torchvision.utils import save_image
import time
import datetime
from torch.utils.data import DataLoader, random_split, SubsetRandomSampler
from torchvision import datasets
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
import torch.nn.functional as F
import torch
from tqdm import tqdm
from Dataset.dataset import *
from Model.Models import *
from torch.fft import fft
import matplotlib.pyplot as plt
import random
import torch.autograd as autograd
from scipy.signal import find_peaks
from scipy.stats.stats import pearsonr
from sklearn.model_selection import KFold
import matplotlib.animation as animation
from Loss import exp_loss
import neurokit2 as nk
import scipy.sparse
#import pingouin as pg
import statsmodels.api as sm
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--rppg', type=str, default='pos', help='rppg Algorithm')
parser.add_argument('--epoch', type=int, default=0,
                    help='epoch to start training from')
parser.add_argument('--weight_path', type=str, default='/edahome/pcslab/pcs04/GAN/PulseGan/Save_model/val_100epochs_32batchsize/', help='path to code and data')
parser.add_argument('--Model_file', type=str, default='/edahome/pcslab/pcs04/RPPG/rPPG/Main/model_input.txt', help='path to Whole input information txt file')
parser.add_argument('--train_epoch', type=int, default=0,
                    help='which pretrained epoch to load')
parser.add_argument('--n_epochs', type=int, default=10,
                    help='number of epochs of training')
parser.add_argument('--Model', type=str, default='LSTM',
                    help='which model to train')
parser.add_argument('--kwidth', type=int, default=7,
                    help='number of kernel width')
parser.add_argument('--n_fc', type=int, default=16,
                    help='number of start kernels')
parser.add_argument('--data_len', type=int, default=300,
                    help='number of data points for one sample')
parser.add_argument('--data_path', type=str,
                    default='/edahome/pcslab/pcs04/RPPG/PulseGan/Datasets', help='path to data')
parser.add_argument('--save_path', type=str,
                    default='/edahome/pcslab/pcs04/RPPG/rPPG/Summary/CNN_auto/', help='path to Save model')
parser.add_argument('--batch_size', type=int, default=64,
                    help='size of the batches')
parser.add_argument('--up_mode', type=str,
                    default='upconv', help='up mode for cnn')
parser.add_argument('--input_size', type=int, default=1,
                    help='input_size for Multi_attn')
parser.add_argument('--emb_size', type=int, default=512,
                    help='emb_size for Multi_attn')
parser.add_argument('--num_layers', type=int, default=6,
                    help='num_layers for Transformer/LSTM')
parser.add_argument('--num_heads', type=int, default=8,
                    help='num_heads for Multi_attn')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout every layers for Multi_attn')
parser.add_argument("--bi", type=bool, default=False,
                    help="Bidirectional for LSTM")                    
parser.add_argument('--n_cpu', type=int, default=8,
                    help='number of cpu threads to use during batch generation')
parser.add_argument('--seed', type=int, default=0, help='seed for random')
parser.add_argument('--study', type=str, default='name', help='study name')
parser.add_argument('--type', type=str, default='diff_Model', help='test type,diff_Model or same_Model')
 
opt = parser.parse_args()
#print(opt)

#kwargs = {'kwidth':[], 'n_fc':[],'data_len':[],'up_mode':'upconv','emb_size':[],'num_layers':[],'num_heads':8,'bi':[],'dropout':[]}

#test_mode = input('Campare whole file or specific model? 1.whole file 2.specific :')

sample_rate = 30.0




def MaxMinNormalization(x):
    x1 = x.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1)).fit(x1)
    out = scaler.transform(x1).reshape(-1)
    # x = (x - Min) / (Max - Min)
    return out


def IBI_cal(signal, sample_rate):
    fre = 1/sample_rate
    #peaks, _ = find_peaks(signal, distance=12, height=0.4)
    info = nk.ppg_findpeaks(signal,sampling_rate=sample_rate)
    peaks = info["PPG_Peaks"]
    
    peak_diff = np.diff(peaks)
    ts = peak_diff*fre  # sec
    IBI = ts.mean()
    HR = 60/IBI
    #HR = nk.signal_rate(peaks=peaks,desired_length=len(signal),sampling_rate=sample_rate).mean()
    return IBI, HR, ts*1000, peaks


def RMSE_cal(gt_HR, HR):

    error = abs(gt_HR-HR)

    squaredError = error * error

    MAE = error.sum()/len(error)
    RMSE = math.sqrt(squaredError.sum() / len(squaredError))
    # print(RMSE)
    return RMSE


def MAE(X, Y):
    error = abs(X-Y)

    MAE = error.sum()/len(error)
    return MAE

def PCC_curve(x,y,name):
    #x:GT,y:estimate
    os.makedirs('{}/{}'.format(opt.save_path, opt.study),exist_ok=True)
    x = np.array(x)
    y = np.array(y)
    
    
    #plt.subplot(1,2,1)
    plt.title('{}'.format(name))
    plt.scatter(x, y)
    #plt.show()
    r1,_ = pearsonr(x,y)
    r1 = (r1)
    a1 = np.polyfit(x, y, 1)
    p1 = np.poly1d(a1)
    plt.plot(np.unique(x), p1(np.unique(x)),label = 'Regression line:{} \n r={}'.format(p1,r1), color='red')
    plt.legend(loc='upper right',fontsize=20)
    plt.xlabel('HR_ground_truth',fontsize=20)
    plt.ylabel('HR_estimated',fontsize=20)
    # plt.subplot(1,2,2)
    # plt.suptitle('HR {}'.format(opt.Model))
    # plt.scatter(z, y)
    # r2,_ = pearsonr(z,y)
    # #plt.show()
    # a2 = np.polyfit(z, y, 1)
    # p2 = np.poly1d(a2)
    # plt.plot(np.unique(z), p2(np.unique(z)),label = 'Regression line:{} \n r={}'.format(p2,r2), color='red')
    
    plt.legend(loc='upper right')
    
    plt.savefig('{}/{}/{}.png'.format(opt.save_path, opt.study,name))
    plt.close()
    return r1,p1
    
def plot_box(data1,data2,name,title_):
    data = []
    data.append(data1[0])
    label = ['POS']
    for index in range(len(data1)):
        data.append(data2[index])
        label.append(name_list[index])
    #x = np.arange(1,len(lable)+1)
    print(label)
    print(data)
    #df = pd.DataFrame(data,columns=label)
    plt.title(title_)
    plt.boxplot(data,labels=label)
    #plt.xticks(x,lable)
    plt.savefig('{}/{}/{}.png'.format(opt.save_path, opt.study,name))
    plt.close()    
    
def test(model, test_loader):

    pea_rate_r, pea_rate_tr = [], []
    MAE_list, MAE_list_tr = [], []
    RMSE_list, RMSE_list_tr = [], []
    HR_AE_list, HR_wave_gt_list, HR_wave_rppg_list = [], [], []
    # IBI_list = []
    # os.makedirs('Fig/{}'.format(opt.sub), exist_ok=True)
    model.eval()
    with torch.no_grad():
        for i, (wave_rppg, wave_gt,exp_ppg_peaks) in enumerate(tqdm(test_loader)):
            #print('i:',i)
            # for i, (wave_rppg_t,wave_gt_t,txt_rppg,txt_ppg) in enumerate(tqdm(test_loader)):
            #if i < 30:
            wave_rppg = Variable(wave_rppg.type(Tensor)).cuda().detach()

            wave_gt = Variable(wave_gt.type(Tensor)).cuda().detach()
            # print(wave_gt.size())
            # wave_gt = wave_gt.type(Tensor).detach()
            output = model(wave_rppg).detach()
            #print(output.size())
            wave_rppg = wave_rppg.squeeze(1).cpu().numpy()
            wave_gt = wave_gt.squeeze(1).cpu().numpy()
            
            output = output.squeeze(1).cpu().numpy()
            
            #print(output.shape)
            # print(len(output[t0:]))
            # print('x:',x.shape)
            # print('y:',y.shape)
            # print('wave_gt',wave_gt.shape)
            # print('output:',output.shape)
            # #print('squeeze:',output.squeeze().shape)
            for k in range(0,output.shape[0]):
                #print('k:{} shape:{}'.format(k,output.shape[0]))
                wave_rppg_ = wave_rppg[k]
                wave_gt_ = wave_gt[k]
                output_ = output[k]            
                #print('output_:',output_.shape)
                rate_rppg, pea_rppg = pearsonr(wave_rppg_, wave_gt_)
                rate_tr, pea_tr = pearsonr(output_, wave_gt_)
                MAE_rppg = MAE(wave_rppg_, wave_gt_)
                MAE_tr = MAE(output_, wave_gt_)
                RMSE_rppg = RMSE_cal(wave_rppg_, wave_gt_)
                RMSE_tr = RMSE_cal(output_, wave_gt_)
                
                pea_rate_r.append(abs(rate_rppg))
                #print('pea_rate:',pea_rate_r)
                pea_rate_tr.append(abs(rate_tr))
                MAE_list.append(MAE_rppg)
                MAE_list_tr.append(MAE_tr)
                RMSE_list.append(RMSE_rppg)
                RMSE_list_tr.append(RMSE_tr)
                #plt.subplot(3,1,1)
                
                # plt.plot(wave_rppg_,label='rppg')
                # plt.plot(wave_gt_,label='ppg')
                # plt.plot(output_,label='DL')
                # plt.legend(loc='upper right')
                # plt.show()
                
                aIBI_AE, HR_AE, peak_diff_AE, peak_AE = IBI_cal(
                    output_, sample_rate)
                # aIBI_AE2,HR_AE2,peak_diff_AE2,peak_AE2 = IBI_cal(AE2,sample_rate)
                aIBI_wave_rppg, HR_wave_rppg, peak_diff_wave_rppg, peak_wave_rppg = IBI_cal(
                    wave_rppg_, sample_rate)
                aIBI_wave_gt, HR_wave_gt, peak_diff_wave_gt, peak_wave_gt = IBI_cal(
                    wave_gt_, sample_rate)
                
                
                #print(HR_AE,HR_wave_rppg,HR_wave_gt)
                # x1 = np.arange(len(wave_gt))
                # plt.plot(x1,wave_gt)
                # plt.plot(peak_wave_gt,wave_gt[peak_wave_gt])
                # plt.show()
                # if aIBI_AE == nan:
                # print(txt_rppg)
                # else if aIBI_wave_rppg == nan:
                # print(txt_rppg)
                HR_AE_list.append(HR_AE)
                # HR_AE_list2.append(HR_AE2)
                HR_wave_rppg_list.append(HR_wave_rppg)
                HR_wave_gt_list.append(HR_wave_gt)
                
            # else:
                # break
            
        HR_AE_list = np.array(HR_AE_list) 
        HR_wave_gt_list = np.array(HR_wave_gt_list)
        HR_wave_rppg_list = np.array(HR_wave_rppg_list)
        
        #######################time signal metrics####################
        pea_rppg = np.array(pea_rate_r)
        #print('all_pea:',pea_rppg)
        pea_ml = np.array(pea_rate_tr)
        MAE_rppg = np.array(MAE_list)
        MAE_ml = np.array(MAE_list_tr)
        RMSE_rppg = np.array(MAE_list)
        RMSE_ml = np.array(MAE_list_tr)
        
        a_pea_r.append(np.mean(pea_rppg))
        a_pea_tr.append(np.mean(pea_ml))
        #print('pearson:',a_pea_r)
        a_MAE_r.append(np.mean(MAE_rppg))
        a_MAE_tr.append(np.mean(MAE_ml))
        a_RMSE_r.append(np.mean(RMSE_rppg))
        a_RMSE_tr.append(np.mean(RMSE_ml))
        
        ####################HR metrics#########################################
        
        MAE_HR_rppg.append(
            MAE(np.array(HR_wave_rppg_list), np.array(HR_wave_gt_list)))
        MAE_HR_AE.append(
            MAE(np.array(HR_AE_list), np.array(HR_wave_gt_list)))
        
        pea_hr_rppg,_ = pearsonr(HR_wave_rppg_list,HR_wave_gt_list)
        pea_hr_AE,_ = pearsonr(HR_AE_list,HR_wave_gt_list)
        
        pea_HR_rppg.append(abs(pea_hr_rppg))        
        pea_HR_AE.append(abs(pea_hr_AE))
        
        RMSE_HR_rppg.append(
            RMSE_cal(np.array(HR_wave_rppg_list), np.array(HR_wave_gt_list)))
        RMSE_HR_AE.append(RMSE_cal(np.array(HR_AE_list),
                          np.array(HR_wave_gt_list)))
        
        
        #print('HR_lens:',len(RMSE_HR_rppg))
        
        out_l.append(output_)
        #print('len pea_rppg:',pea_rppg.shape)
        #print('len a_pea_r:',len(a_pea_r))
        
        #peak_rppg.append(peak_wave_rppg)
        #peak_ppg.append(peak_wave_gt)
    return a_pea_r, a_pea_tr, a_MAE_r, a_MAE_tr,a_RMSE_r,a_RMSE_tr,MAE_HR_rppg,MAE_HR_AE,pea_HR_rppg,pea_HR_AE, RMSE_HR_rppg, RMSE_HR_AE, out_l, wave_rppg_, wave_gt_, output_,HR_AE_list,HR_wave_rppg_list,HR_wave_gt_list,pea_rppg,pea_ml,MAE_rppg,MAE_ml,RMSE_rppg,RMSE_ml

def load_parm(p_path):
    #pos_Transformer_emb256_lr0001_b8_k7_layers6_newlossbeta05_newpeak_drop02_d1
    
    if '_k' in p_path:
        x = p_path.index('_k')
        print('k:',p_path[x:x+5])
        kwidth = int(''.join([y for y in p_path[x:x+5] if y.isdigit()])) 
    else:
        kwidth = opt.kwidth
    
    if 'emb' in p_path:        
        x = p_path.index('emb')
        print(p_path[x:x+7])
        emb_size = int(''.join([y for y in p_path[x:x+7].split('_')[0] if y.isdigit()])) 
    elif '_h' in p_path:
        x = p_path.index('_h')
        print(p_path[x:x+6])
        emb_size = int(''.join([y for y in p_path[x:x+6] if y.isdigit()])) 

    else:
        emb_size = opt.emb_size
   
    if 'n32' in p_path:        
        n_fc = 32
    elif 'n64' in p_path:        
        n_fc = 64    
    elif 'n16' in p_path:        
        n_fc = 16    
    elif 'n8' in p_path:
        n_fc = 8
    else:
        n_fc = opt.n_fc
        
    if 'layers' in p_path:
        x = p_path.index('layers')
        print(p_path[x:x+8])
        num_layers = int(''.join([y for y in p_path[x:x+8] if y.isdigit()]))             
    else:
        num_layers = opt.num_layers  
    
    if 'heads' in p_path:
        x = p_path.index('heads')
        num_heads = int(''.join([y for y in p_path[x-2:x] if y.isdigit()]))             
    else:
        num_heads = opt.num_heads
    
    if 'bi' in p_path:
        bi = bool(1)        
    else:
        bi = bool(0)  
    
    data_len = opt.data_len
    up_mode = opt.up_mode    

        
    return kwidth, n_fc,  data_len, up_mode, emb_size, num_layers, num_heads, bi
    
def load_model(type):
    

    if type == 'diff_Model':   
        name_list,model_list,ckp_list,kwidth_list,n_fc_list,data_len_list,emb_size_list,num_layers_list,num_heads_list,bi_list = [],[],[],[],[],[],[],[],[],[]
        
        with open(opt.Model_file,'r') as file:
            lines = file.read().splitlines()
            #print(lines)
            l_num = 0
            for index in lines:
                if l_num%3 == 0:
                    model_list.append(index)    
                elif l_num%3 == 1:
                    ckp_list.append(index)
                else:
                    name_list.append(index)
                #print(i%3)
                #print(index)
                l_num += 1
                
        print(model_list)
        #finish = 1
        #while finish == 1:
        #Model = int(input('Model:1.CNN_auto 2.LSTM 3.GAN 4.Transformer :'))
        #model_list.append(Model)
        #path_ = input('input cpt path :')
        #state_ = torch.load(path_)
        #ckp_list.append(path_)
        #sub_name = str(input('sub_name:'))
        #name_list.append(sub_name)
        
        #print(type(state_))
        for path_ in ckp_list:
            state_ = torch.load(path_)
            if state_.get('kwargs') :
                
                kwidth_list.append(0) 
                n_fc_list.append(0)
                data_len_list.append(0)
                emb_size_list.append(0)
                num_layers_list.append(0)
                num_heads_list.append(0)
                bi_list.append(0)
                
                #finish = int(input('Continue typing? 1.Yes 2.No :'))
                    
            else:
                #print('type required parameters,if the parameter you don\'t need,just input 0:')
                
                parm_path = os.path.dirname(path_).split('/')[-1]
                print(parm_path)
                kwidth, n_fc,  data_len, up_mode, emb_size, num_layers, num_heads, bi = load_parm(parm_path)       
                
                 # = int(input('number of CNN start kernels :'))
                
                 # = int(input('data lens :'))
                
                # #up_mode = str(input())
                # emb_size = int(input('emb_size for Transformer or hidden_size for LSTM :'))
                
                # num_layers = int(input('number of layers :'))
                
                # num_heads = int(input('number of heads :'))
                
                # bi = bool(input('bidirectional for LSTM ? 0.False 1.True :'))        
                #dropout = input('Dropout:')
                # if dropout != ' ':
                    # dropout = float(dropout)
                
                kwidth_list.append(kwidth) 
                n_fc_list.append(n_fc)
                data_len_list.append(data_len)
                emb_size_list.append(emb_size)
                num_layers_list.append(num_layers)
                num_heads_list.append(num_heads)
                bi_list.append(bi)
                #dropout_list.append(dropout)    
                
                #print('kwidth:',kwidth,' n_fc:',n_fc,' data_len',data_len,' emb_size:',emb_size,' num_layers:',num_layers,' bidirectional:',bi,' dropout:',dropout)
                #print(kwargs)
                #k = input('kwidth(old int number) :')
                #print(k)
                #if Model == 1:
                
                #finish = int(input('Continue typing? 1.Yes 2.No :'))
    
    else:
        name_list = 0
        model_list = opt.Model
        ckp_list = 0
        kwidth_list = opt.kwidth
        n_fc_list = opt.kwidth
        data_len_list = 300
        emb_size_list = opt.emb_size
        num_layers_list = opt.num_layers
        num_heads_list = opt.num_heads
        bi_list = opt.bi
        
        
    
    return name_list,model_list,ckp_list,kwidth_list,n_fc_list,data_len_list,emb_size_list,num_layers_list,num_heads_list,bi_list

def analysing(type):
    ##DL Model Parameters###
    if type == 'diff_Model':
        for idx in range(len(model_list)): 
            Model = model_list[idx]
            
            state = torch.load(ckp_list[idx])
            epoch = state['epoch']
            print(epoch)
            
            if state.get('kwargs'):
                #print('yes')
                kwargs = state['kwargs']
            else:            
                kwargs = {'input_size':1,'kwidth':kwidth_list[idx], 'n_fc':n_fc_list[idx],'data_len':data_len_list[idx],'up_mode':'upconv','emb_size':emb_size_list[idx],'num_layers':num_layers_list[idx],'num_heads':num_heads_list[idx],'bi':bi_list[idx],'dropout':0}
                #print('no')
                
            #kwargs = {'input_size':1,'kwidth':kwidth_list[idx], 'n_fc':n_fc_list[idx],'data_len':data_len_list[idx],'up_mode':'upconv','emb_size':opt.emb_size,'num_layers':opt.num_layers,'num_heads':num_heads_list[idx],'bi':opt.bi,'dropout':0.2}
            #print(type(bi_list[idx]))
            #print(type(opt.bi))
            print(kwargs)
            
            
            if Model == 'CNN_auto':
                model = CNN_auto(**kwargs)
            elif Model == 'ECG_auto':
                model = PPG2ECG(**kwargs)
            elif Model == 'LSTM':
                model = RNN_model(**kwargs)
            elif Model == 'LSTM_auto':
                model = PPG2ECG_BASELINE_LSTM(**kwargs)
            elif Model == 'GAN':
                model = Generator(**kwargs)

            elif Model == 'Transformer':
                model = Trans(**kwargs)
            else:
                model = CNN_auto(**kwargs)
            
            
            #kwargs = state['kwargs']
            #print(kwargs)
            #optimizer.load_state_dict(state_G['optimizer'])

            #if int(epoch) <= 50:
            
            model.load_state_dict(state['state_dict'])
            print(model)
            epoch_list.append(epoch)
            if cuda:
                model=model.cuda()
            #RMSE_list, epoch_list = [], []


            a_pea_r, a_pea_tr, a_MAE_r, a_MAE_tr,a_RMSE_r,a_RMSE_tr,MAE_HR_rppg,MAE_HR_AE,pea_HR_rppg,pea_HR_AE, RMSE_HR_rppg, RMSE_HR_AE, out_l, wave_rppg_, wave_gt_, output_,HR_AE_list,HR_wave_rppg_list,HR_wave_gt_list,pea_rppg,pea_ml,MAE_rppg,MAE_ml,RMSE_rppg,RMSE_ml = test(
                            model, test_loader)
                    
            
            summary_signal['pea_rppg'].append(pea_rppg)
            summary_signal['pea_ml'].append(pea_ml)
            summary_signal['MAE_rppg'].append(MAE_rppg)
            summary_signal['MAE_ml'].append(MAE_ml)
            summary_signal['RMSE_rppg'].append(RMSE_rppg)
            summary_signal['RMSE_ml'].append(RMSE_ml)
            
            
            summary_HR['ml'].append(HR_AE_list)
            summary_HR['rppg'].append(HR_wave_rppg_list)
            summary_HR['gt'].append(HR_wave_gt_list)
            

            
            #print('pea_len:',len(a_pea_r))
            ri,pi = PCC_curve(HR_wave_gt_list,HR_AE_list,'HR_MAE_{}'.format(name_list[idx]))
            
            r_HR.append(ri)
            p_HR.append(pi)
            print('a_pea_tr:',a_pea_tr)
    else:
        
        Model = opt.Model
        kwargs = {'input_size':1,'kwidth':kwidth_list, 'n_fc':n_fc_list,'data_len':data_len_list,'up_mode':'upconv','emb_size':emb_size_list,'num_layers':num_layers_list,'num_heads':num_heads_list,'bi':bi_list,'dropout':0}
        print(kwargs)
        
        if opt.Model == 'CNN_auto':
            model = CNN_auto(**kwargs)
        elif opt.Model == 'ECG_auto':
            model = PPG2ECG(**kwargs)
        elif opt.Model == 'LSTM':
            model = RNN_model(**kwargs)
        elif opt.Model == 'LSTM_auto':
            model = PPG2ECG_BASELINE_LSTM(**kwargs)
        elif opt.Model == 'GAN':
            model = Generator(**kwargs)

        elif opt.Model == 'Transformer':
            model = Trans(**kwargs)
            
        for dirPath, dirNames, fileNames in os.walk(opt.weight_path):
            for name in fileNames:
                state = torch.load(os.path.join(opt.weight_path,name))
                epoch = state['epoch']
                
                print(epoch)
                #if int(epoch) <= 100:
            
                
                model.load_state_dict(state['state_dict'])
                print(model)
                epoch_list.append(epoch)
                if cuda:
                    model=model.cuda()
                #RMSE_list, epoch_list = [], []


                a_pea_r, a_pea_tr, a_MAE_r, a_MAE_tr,a_RMSE_r,a_RMSE_tr,MAE_HR_rppg,MAE_HR_AE,pea_HR_rppg,pea_HR_AE, RMSE_HR_rppg, RMSE_HR_AE, out_l, wave_rppg_, wave_gt_, output_,HR_AE_list,HR_wave_rppg_list,HR_wave_gt_list,pea_rppg,pea_ml,MAE_rppg,MAE_ml,RMSE_rppg,RMSE_ml = test(
                        model, test_loader)
                
                # summary_signal['a_pea_rppg'].append(a_pea_r)
                # summary_signal['a_pea_ml'].append(a_pea_tr)
                # summary_signal['a_MAE_rppg'].append(a_MAE_r)
                # summary_signal['a_MAE_ml'].append(a_MAE_tr)
                # summary_signal['a_RMSE_rppg'].append(a_RMSE_r)
                # summary_signal['a_RMSE_ml'].append(a_RMSE_tr) 
                # summary_signal['pea_rppg'].append(pea_rppg)
                # summary_signal['pea_ml'].append(pea_ml)
                # summary_signal['MAE_rppg'].append(MAE_rppg)
                # summary_signal['MAE_ml'].append(MAE_ml)
                # summary_signal['RMSE_rppg'].append(RMSE_rppg)
                # summary_signal['RMSE_ml'].append(RMSE_ml)
                
                # summary_HR['a_pea_rppg'].append(pea_HR_rppg)
                # summary_HR['a_pea_ml'].append(pea_HR_AE)
                # summary_HR['a_MAE_rppg'].append(MAE_HR_rppg)
                # summary_HR['a_MAE_ml'].append(MAE_HR_AE)
                # summary_HR['a_RMSE_rppg'].append(RMSE_HR_rppg)
                # summary_HR['a_RMSE_ml'].append(RMSE_HR_AE) 
                #print('pea_len:',len(a_pea_r))
                ri,pi = PCC_curve(HR_wave_gt_list,HR_AE_list,'HR_MAE_{}'.format(epoch))
                
                r_HR.append(ri)
                p_HR.append(pi)
                #print('a_pea_tr:',a_pea_tr)
                #print('HR_RMSE:',(RMSE_HR_rppg, RMSE_HR_AE))
            
    ri,pi = PCC_curve(HR_wave_gt_list,HR_wave_rppg_list,'HR_MAE_rPPG')
    return 
test_dataset = TrainDataset(path=opt.data_path, mode='test',
                            rppg=opt.rppg, data_len=opt.data_len)

#train_set_size = int(len(test_dataset) * 0.8)
#valid_set_size = len(test_dataset) - train_set_size
# test_dataset, validation_dataset = torch.utils.data.random_split(
    # test_dataset, [train_set_size, valid_set_size])

test_loader = DataLoader(
    # train_d,
    test_dataset,
    # rppg = 'xminay',
    batch_size=opt.batch_size,
    shuffle=False,
    num_workers=0,

)

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


   
    
        
    #print(test_mode)
    #print(ckp_list)
    
print('*********************************Start analysing****************************')          

####load model####
'''
for i in range(len(model_list)):
    model,epoch = load_model(model_list[i],i,ckp_list)
    print('model:',model)
    print('epoch:',epoch)


print(ckp_list)
'''



a_pea_r, a_pea_tr = [], []
a_MAE_r, a_MAE_tr = [], []
a_RMSE_r, a_RMSE_tr = [], []
a_RMSE, a_RMSE_fake = [], []
RMSE_HR_rppg, RMSE_HR_AE = [], []
MAE_HR_rppg, MAE_HR_AE = [], []
pea_HR_rppg, pea_HR_AE = [], []
peak_rppg,peak_ppg = [],[]
out_l = []
epoch_list = []
r_HR,p_HR = [],[]
train_loss, train_loss_G, train_loss_D = [], [], []
eval_loss = []

#a_pea_r, a_pea_tr, a_MAE_r, a_MAE_tr,a_RMSE_r,a_RMSE_tr,MAE_HR_rppg,MAE_HR_AE,pea_HR_rppg,pea_HR_AE, 
#RMSE_HR_rppg, RMSE_HR_AE, out_l, wave_rppg_, wave_gt_, output_,HR_AE_list,HR_wave_rppg_list,HR_wave_gt_list
#,pea_rppg,pea_ml,MAE_rppg,MAE_ml,RMSE_rppg,RMSE_ml

summary_signal = {'pea_rppg':[],'pea_ml':[],'MAE_rppg':[],'MAE_ml':[],'RMSE_rppg':[],'RMSE_ml':[]}
summary_HR = {'ml':[],'rppg':[],'gt':[]}
name_list,model_list,ckp_list,kwidth_list,n_fc_list,data_len_list,emb_size_list,num_layers_list,num_heads_list,bi_list = load_model(opt.type)

analysing(opt.type) 
plot_box(summary_signal['pea_rppg'],summary_signal['pea_ml'],'PCC_box','PCC')
plot_box(summary_signal['MAE_rppg'],summary_signal['MAE_ml'],'MAE_box','MAE')


with open('{}/{}.txt'.format(opt.save_path, opt.study), 'w') as file: 
    file.write("PCC line epoch"+"\n")
    for i in range((len(r_HR))):
        file.write(str(r_HR[i]) + " " + str(p_HR[i]) + " " + str(epoch_list[i])
                   + "\n")

for i in range(0,len(name_list)):
    print(name_list[i])
    print('Signal')
    print('MAE:{} RMSE:{} PCC:{}'.format(a_MAE_tr[i],a_RMSE_tr[i],a_pea_tr[i]))
    print('HR')
    print('MAE:{} RMSE:{} PCC:{}'.format(MAE_HR_AE[i],RMSE_HR_AE[i],pea_HR_AE[i]))

print('POS')
print('Signal')
print('MAE:{} RMSE:{} PCC:{}'.format(a_MAE_r[0],a_RMSE_r[0],a_pea_r[0]))
print('HR')
print('MAE:{} RMSE:{} PCC:{}'.format(MAE_HR_rppg[0],RMSE_HR_rppg[0],pea_HR_rppg[0]))
    
# for dirPath, dirNames, fileNames in os.walk(opt.weight_path): 
   
    # #if iter < 2:
        # #print(iter)
    # for name in fileNames:
        # print(os.path.join(dirPath,name))

'''
state = torch.load(opt.weight_path)
epoch = state['epoch']
print(epoch)

#optimizer.load_state_dict(state_G['optimizer'])

#if int(epoch) <= 50:
model.load_state_dict(state['state_dict'])
print(model)
epoch_list.append(epoch)
if cuda:
    model=model.cuda()
#RMSE_list, epoch_list = [], []


a_pea_r, a_pea_tr, a_MAE_r, a_MAE_tr,a_RMSE_r,a_RMSE_tr,MAE_HR_rppg,MAE_HR_AE,pea_HR_rppg,pea_HR_AE, RMSE_HR_rppg, RMSE_HR_AE, out_l, wave_rppg_, wave_gt_, output,HR_AE_list,HR_wave_rppg_list,HR_wave_gt_list,pea_rppg,pea_ml,MAE_rppg,MAE_ml = test(
        model, test_loader)
#print('pea_len:',len(a_pea_r))

print('pea:',(pea_HR_rppg,pea_HR_AE))
print('HR_RMSE:',(RMSE_HR_rppg, RMSE_HR_AE))
'''
#iter = iter + 1
    # else:
        # continue



# #HR_AE_list,HR_wave_rppg_list,HR_wave_gt_list
# fig, axes = plt.subplots()
# #ax1 = axes[0]
# #ax2 = axes[1]
# sm.graphics.mean_diff_plot(RMSE_HR_AE, RMSE_HR_rppg, ax = axes)
# #sm.graphics.mean_diff_plot(HR_wave_rppg_list, HR_wave_gt_list, ax = ax2)
# plt.show()

# lable = ['rPPG','DL']
# x = np.arange(1,len(lable)+1)

# data_df = [MAE_rppg,MAE_ml]
            
# plt.title("MAE")
# plt.boxplot(data_df)
# plt.xticks(x,lable)
# plt.savefig('{}/{}/{}.png'.format(opt.save_path, opt.study,'boxplot'))
# plt.close()
# #plt.show()
# PCC_curve(HR_wave_rppg_list,HR_wave_gt_list,HR_AE_list)
