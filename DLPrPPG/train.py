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
parser = argparse.ArgumentParser()
parser.add_argument('--rppg', type=str, default='pos', help='rppg Algorithm')
parser.add_argument('--epoch', type=int, default=0,
                    help='epoch to start training from')
# parser.add_argument('--weight_path', type=str, default='/edahome/pcslab/pcs04/GAN/PulseGAN/Save_model/val_100epochs_32batchsize/', help='path to code and data')
parser.add_argument('--train_epoch', type=int, default=0,
                    help='which pretrained epoch to load')
parser.add_argument('--n_epochs', type=int, default=50,
                    help='number of epochs of training')
parser.add_argument('--Model', type=str, default='LSTM',
                    help='which model to train:CNN_auto,LSTM,GAN,Transformer')
parser.add_argument('--GAN', type=str, default='CGAN',
                    help='CGAN,WGAN_GP')
parser.add_argument('--kwidth', type=int, default=7,
                    help='number of kernel width')
parser.add_argument('--n_fc', type=int, default=16,
                    help='number of start kernels')
parser.add_argument('--data_len', type=int, default=300,
                    help='number of data points for one sample')
parser.add_argument('--data_path', type=str,
                    default='/edahome/pcslab/pcs04/RPPG/PulseGan/Datasets', help='path to data')
parser.add_argument('--save_model_path', type=str,
                    default='/edahome/pcslab/pcs04/RPPG/Transformer/CNN/', help='path to Save model')
parser.add_argument('--target_wave', type=str, default='saw',
                    help='generator signal:sine,cos,square,saw')
parser.add_argument('--gt', type=str, default='sine', help='ground_truth')
parser.add_argument('--up_mode', type=str,
                    default='upconv', help='up mode for cnn')
parser.add_argument('--batch_size', type=int, default=64,
                    help='size of the batches')
parser.add_argument('--input_size', type=int, default=1,
                    help='input_size for Multi_attn')
parser.add_argument('--emb_size', type=int, default=512,
                    help='emb_size for Multi_attn')
parser.add_argument('--num_layers', type=int, default=3,
                    help='num_layers for Transformer/LSTM')
parser.add_argument('--num_heads', type=int, default=8,
                    help='num_heads for Multi_attn')
parser.add_argument("--lr", type=float, default=0.0001,
                    help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5,
                    help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999,
                    help="adam: decay of first order momentum of gradient")
parser.add_argument('--amp', type=int, default=1, help='parameter of amp')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout every layers for Multi_attn')
parser.add_argument("--SGD", type=bool, default=False,
                    help="Use SGD for optimizer")
parser.add_argument("--bi", type=bool, default=False,
                    help="Bidirectional for LSTM")                    
# parser.add_argument('--alpha', type=int, default=10, help='parameter of time loss')
parser.add_argument("--Dis", type=str, default='normal',
                    help="use WGAN_GP,SN,normal for D")
parser.add_argument("--dis_loss", type=str, default='hinge',
                    help="loss for D:WGAN_GP or hinge")
parser.add_argument('--beta', type=float, default=0.5, help='parameter exp_loss')
# parser.add_argument('--decay_epoch', type=int, default=50, help='epoch from which to start lr decay')
parser.add_argument('--n_cpu', type=int, default=8,
                    help='number of cpu threads to use during batch generation')
parser.add_argument('--sample_interval', type=int, default=1,
                    help='interval between sampling of images from generators')
parser.add_argument('--seed', type=int, default=0, help='seed for random')
parser.add_argument("--n_critic", type=int, default=5,
                    help="number of training steps for discriminator per iter")
parser.add_argument("--clip_value", type=float, default=0.01,
                    help="lower and upper clip value for disc. weights")
parser.add_argument('--padding_mode', type=str, default='zeros',
                    help='zeros, reflect, replicate or circular')
parser.add_argument('--Loss', type=str, default='L1', help='L1 or L2')
parser.add_argument('--study', type=str, default='name', help='study name')

opt = parser.parse_args()
print(opt)

###DL Model Parameters###
kwargs = {'input_size':opt.input_size,'kwidth':opt.kwidth, 'n_fc':opt.n_fc,'data_len':opt.data_len,'up_mode':opt.up_mode,'emb_size':opt.emb_size,'num_layers':opt.num_layers,'num_heads':opt.num_heads,'bi':bool(opt.bi),'dropout':opt.dropout}
#kwargs = {'input_size':1,'kwidth':7, 'n_fc':16,'data_len':300,'up_mode':'upconv','emb_size':512,'num_layers':6,'num_heads':8,'bi':True,'dropout':0.2}

sample_rate = 30.0

# Loss weight for gradient penalty
lambda_gp = 10

def IBI_cal(signal, sample_rate):
    fre = 1/sample_rate
    # if opt.Model != ('cGAN' or 'WGAN_GP'):
        # peaks, _ = find_peaks(signal, distance=12, height=0.4)
        # info = nk.ppg_findpeaks(signal,sampling_rate=sample_rate)
        # peaks = info["PPG_Peaks"]
    # else:
    peaks, _ = find_peaks(signal, distance=12, height=0.4)
        
    peak_diff = np.diff(peaks)
    ts = peak_diff*fre  # sec
    IBI = ts.mean()
    HR = 60/IBI
    #HR = nk.signal_rate(peaks=peaks,desired_length=len(signal)).mean()
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


def normalize_gradient(net_D, x, rppg):
    """
                     f
    f_hat = --------------------
            || grad_f || + | f |
    """
    x.requires_grad_(True)
    f = net_D(x,rppg)
    grad = torch.autograd.grad(
        f, [x], torch.ones_like(f), create_graph=True, retain_graph=True)[0]
    grad_norm = torch.norm(torch.flatten(grad, start_dim=1), p=2, dim=1)
    grad_norm = grad_norm.view(-1, *[1 for _ in range(len(f.shape) - 1)])
    f_hat = (f / (grad_norm + torch.abs(f)))
    return f_hat

def compute_gradient_penalty(D, real_samples, fake_samples, rppg):
    """Calculates the gradient penalty loss for WGAN GP.
       Warning: It doesn't compute the gradient w.r.t the labels, only w.r.t
       the interpolated real and fake samples, as in the WGAN GP paper.
    """
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1)))
    # print(alpha.size())
    wave_rppg = Tensor(rppg)
    # print(wave_rppg)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha)
                    * fake_samples)).requires_grad_(True)
    # print(real_samples.size())
    # print(interpolates.size())
    d_interpolates = D(interpolates, wave_rppg)
    fake = Tensor(real_samples.shape[0], 1).fill_(1.0)
    fake.requires_grad = False
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )
    # print(gradients[0].size(0))
    gradients = gradients[0].view(gradients[0].size(0), -1)
    # print(((gradients.norm(2, dim=1) - 1) ** 2).size())
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def weight_init(m):
    # weight_initialization: important for wgan
    class_name=m.__class__.__name__
    if class_name.find('Conv')!=-1:
        m.weight.data.normal_(0,0.02)
    elif class_name.find('Batch')!=-1:
        m.weight.data.normal_(1.0,0.02)

cuda = True if torch.cuda.is_available() else False
print('seed:', opt.seed)
# seed initialization
torch.manual_seed(opt.seed)
torch.cuda.manual_seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)  # if you are using multi-GPU.
np.random.seed(opt.seed)  # Numpy module.
random.seed(opt.seed)  # Python random module.
torch.manual_seed(opt.seed)

torch.backends.cudnn.benchmark = False
# torch.set_deterministic(True)
torch.backends.cudnn.deterministic = True
#torch.use_deterministic_algorithms(True)

if opt.Loss == 'L1':
    criterion = torch.nn.L1Loss()  # L1-loss
elif opt.Loss == 'L2':
    criterion = torch.nn.MSELoss()
elif opt.Loss == 'exp':
    criterion = exp_loss(beta=opt.beta)

if  opt.Model == 'cGAN':

    criterion_GAN = torch.nn.MSELoss()#L2-loss
    criterion_freq = torch.nn.L1Loss()

# Initialize Models
if opt.Model == 'CNN_auto':
    model = CNN_auto(**kwargs)
elif opt.Model == 'ECG_auto':
    model = PPG2ECG(**kwargs)
elif opt.Model == 'LSTM':
    model = RNN_model(**kwargs)
elif opt.Model == 'LSTM_auto':
    model = PPG2ECG_BASELINE_LSTM(**kwargs)
elif opt.Model == 'cGAN':
    model = Generator(**kwargs)
elif  opt.Model == 'WGAN_GP':
    model = Generator(**kwargs)
elif opt.Model == 'Transformer':
    model = Trans(**kwargs)

if opt.Model == 'cGAN' or opt.Model == 'WGAN_GP':
    print('use GAN')
    if opt.Dis == 'WGAN_GP':
        print('gp_D')
        discriminator = Discriminator_gp(**kwargs)
    elif opt.Dis == 'SN':
        print('SN')
        discriminator = Discriminator_SN(**kwargs)
    else:
        print('normal')
        discriminator = Discriminator(**kwargs)

        
        
print('{}:'.format(opt.Model), model)

if opt.Model == 'cGAN' or opt.Model == 'WGAN_GP':
    print('Discriminator:', discriminator)
    discriminator.apply(weight_init)
    model.apply(weight_init)

# input_size=1,kernel_size=3,emb_size=512,num_heads=8, num_layers=3,dropout=0.2

if cuda:
    model.cuda()
    criterion.cuda()
    if opt.Model == 'cGAN' or opt.Model == 'WGAN_GP':
        discriminator.cuda()
        
    if opt.Model == 'cGAN':
        criterion_GAN.cuda()
        criterion_freq.cuda()

# Optimizers
optimizer = optim.Adam(model.parameters(), lr=opt.lr)
scheduler = ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3, verbose=True)
if opt.Model == 'cGAN' or opt.Model == 'WGAN_GP':
    
    optimizer1 = optim.Adam(discriminator.parameters(),
                            lr=opt.lr)
    scheduler1 = ReduceLROnPlateau(
        optimizer1, mode='min', factor=0.5, patience=3, verbose=True)
    print('use optimizer1: ',optimizer1)
# kf = KFold(n_splits=5, random_state=None, shuffle=False)
# print(kf)

########load data################
train_dataset = TrainDataset(path=opt.data_path, mode='train',
                             rppg=opt.rppg, data_len=opt.data_len)
print(len(train_dataset))
train_set_size = int(len(train_dataset) * 0.8)
valid_set_size = len(train_dataset) - train_set_size
train_dataset, validation_dataset = torch.utils.data.random_split(
    train_dataset, [train_set_size, valid_set_size])
print('train:', len(train_dataset))
print('val:', len(validation_dataset))
# train_dataset,validation_dataset = torch.utils.data.random_split(train_dataset,[3185, 350])
test_dataset = TrainDataset(path=opt.data_path, mode='test',
                            rppg=opt.rppg, data_len=opt.data_len)


train_loader = DataLoader(
    # train_d,
    train_dataset,
    # rppg = 'xminay',
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=0,
    # drop_last=True
)

val_loader = DataLoader(
    # train_d,
    validation_dataset,
    # rppg = 'xminay',
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=0,
    # drop_last=True
)

test_loader = DataLoader(
    # train_d,
    test_dataset,
    # rppg = 'xminay',
    batch_size=128,
    shuffle=False,
    num_workers=0,

)


# print('len of train batch is: ', len(train_loader))
# print('len of val batch is: ', len(val_loader))

# Tensor type
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor
# Tensor = torch.FloatTensor
# LongTensor = torch.LongTensor


prev_time = time.time()
print(prev_time)


def train(model, train_loader, epoch):
    # train except GAN
    model.train()
    train_loss = 0
    n = 0
    for i, (wave_rppg, wave_gt,exp_ppg_peaks) in enumerate(tqdm(train_loader)):
        batch_size = wave_rppg.shape[0]
        #noise = torch.rand(wave_rppg.shape)
        #print(type(wave_rppg),type(exp_ppg_peaks))
        #print(wave_rppg.size(),exp_ppg_peaks.size())
        # Model input
        # print('wave_rppg',wave_rppg.size())#(batch,channel,lens)
        # real_data
        wave_rppg = Variable(wave_rppg.type(Tensor)).cuda()
        #noise = Variable(noise.type(Tensor)).cuda()
        # print(wave_rppg.size())
        wave_gt = Variable(wave_gt.type(Tensor)).cuda()
        exp_ppg_peaks = Variable(exp_ppg_peaks.type(Tensor)).cuda()
        # print('wave_gt:',wave_gt.size())#(batch,channel,lens)
        #print(wave_rppg)
        optimizer.zero_grad()

        output = model(wave_rppg)

        # print('output:',output.size())#(batch,channel,lens)

        # Loss for real
        # print(output.squeeze().size(),wave_gt.squeeze().size())
        if opt.Loss == 'exp':
            loss = criterion(output, wave_gt,exp_ppg_peaks)
        else:    
            loss = criterion(output, wave_gt)
        # print(output.squeeze().size())
        loss.backward()
        optimizer.step()
        # print(loss.item())
        train_loss += (loss.detach().cpu().item() * batch_size)
        # print('train_loss:',train_loss)
        n += batch_size
        # print('n/train_loader:{}/{}'.format(n,len(train_loader)))
        # print()
        
    #################save model###################################################

    return train_loss

def train_WGAN(model, discriminator, train_loader, epoch, sample_interval):
    # train except GAN
    model.train()
    discriminator.train()
    train_loss = 0
    #G_train_loss_his, D_train_loss_his = [], []
    #RMSE_list, epoch_list = [], []
    G_runing_loss = 0
    D_runing_loss = 0

    n = 0
    for kk, (wave_rppg, wave_gt,exp_ppg_peaks) in enumerate(tqdm(train_loader)):
        batch_size = wave_rppg.shape[0]
        #print(wave_rppg.shape)
        #print(wave_rppg.shape)
        # Model input
        # print('wave_rppg',wave_rppg.size())#(batch,channel,lens)
        # real_data
        #noise = torch.rand(wave_rppg.shape)
        wave_rppg = Variable(wave_rppg.type(Tensor)).cuda()
        # print(wave_rppg.size())
        #noise = Variable(noise.type(Tensor)).cuda()
        wave_gt = Variable(wave_gt.type(Tensor)).cuda()
        # print('wave_gt:',wave_gt.size())#(batch,channel,lens)
                    
        optimizer1.zero_grad()
        #print(model)
        #print(wave_rppg)
        output = model(wave_rppg)
        #print(output)
        # print('output:',output.size())#(batch,channel,lens)

        # Loss for real
        if opt.dis_loss == 'GN':
            validity_real = normalize_gradient(discriminator, wave_gt, wave_rppg)
            validity_fake = normalize_gradient(discriminator, output, wave_rppg)
            #print('GN')
        else:
            validity_real = discriminator(wave_gt, wave_rppg )
            validity_fake = discriminator(output, wave_rppg )
            gradient_penalty = compute_gradient_penalty(
                discriminator, wave_gt.data, output.data, wave_rppg)
        # print('validity_size:',validity_real.size())
        # print('real: {} fake: {}'.format(validity_real,validity_fake))

        if opt.dis_loss == 'hinge':
            D_Loss = torch.mean(nn.ReLU()(1.0 - validity_real)) + torch.mean(nn.ReLU()(1.0 + validity_fake))
                
        elif opt.dis_loss == 'WGAN_GP':
            #print('gp_loss')
            D_Loss = -torch.mean(validity_real) + torch.mean(validity_fake) + lambda_gp * gradient_penalty
                
        elif opt.dis_loss == 'GN':
            D_Loss = -torch.mean(validity_real) + torch.mean(validity_fake)
        
        #print(gradient_penalty)
        # print('Loss_size:',D_Loss)
        # print(D_Loss)
        D_Loss.backward()
        optimizer1.step()
        n += batch_size

        # Train the generator every n_critic iterations
        if kk % opt.n_critic == 0:

            # -----------------
            #  Train Generator
            # -----------------
            optimizer.zero_grad()

            gen_rppg = model(wave_rppg)
            if opt.dis_loss == 'GN':
                pred_fake = normalize_gradient(discriminator, gen_rppg, wave_rppg)
            else:
                pred_fake = discriminator(gen_rppg, wave_rppg )

            G_loss = -torch.mean(pred_fake)

            G_loss.backward()

            optimizer.step()

        G_runing_loss += G_loss.item() * batch_size
        D_runing_loss += D_Loss.item() * batch_size

    # G_runing_loss = G_runing_loss/len(train_loader)
    # D_runing_loss = D_runing_loss/len(train_loader)

        # print('n/train_loader:{}/{}'.format(n,len(train_loader)))
        # print()
    if (epoch) % sample_interval == 0:
        print(sample_interval, epoch)
        s_path = os.path.join(opt.save_model_path, 'cpt', opt.study)
        os.makedirs(s_path, exist_ok=True)
        
        state_G = {'epoch': epoch, 'state_dict': model.state_dict(),
                   'optimizer': optimizer.state_dict(),'kwargs': kwargs}
        torch.save(state_G, os.path.join(s_path, 'save_{}.pth'.format((epoch))))
        # state_D = {'epoch': epoch, 'state_dict': discriminator.state_dict(),
                   # 'optimizer': optimizer1.state_dict()}
        # torch.save(state_D, os.path.join(s_path, 'D_{}.pth'.format((epoch))))
    return G_runing_loss, D_runing_loss



def train_GAN(model, discriminator, train_loader, epoch, sample_interval):
    # train except GAN
    model.train()
    discriminator.train()
    train_loss = 0
    #G_train_loss_his, D_train_loss_his = [], []
    #RMSE_list, epoch_list = [], []
    G_runing_loss = 0
    D_runing_loss = 0

    n = 0
    for i, (wave_rppg, wave_gt,exp_ppg_peaks) in enumerate(tqdm(train_loader)):
        batch_size = wave_rppg.shape[0]
        # Model input
        # print('wave_rppg',wave_rppg.size())#(batch,channel,lens)
        # real_data
        wave_rppg = Variable(wave_rppg.type(Tensor)).cuda()
        # print(wave_rppg.size())
        wave_gt = Variable(wave_gt.type(Tensor)).cuda()
        # print('wave_gt:',wave_gt.size())#(batch,channel,lens)
        
        
        valid = Variable(Tensor(batch_size, 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(batch_size, 1).fill_(0.0), requires_grad=False)
        valid_smoothed = torch.add(valid, -0.1)
        #print('valid:',valid.size())
        
        # -----------------
        #  Train Generator
        # ----------------- 
        
        
        optimizer.zero_grad()
        #z = Variable(Tensor(np.random.normal(0.5, 0.5, (batch_size,1,opt.data_len))))
        #print(wave_rppg.size())
        #print(z.size())
        fake_rppg = model(wave_rppg)
        pred_fake = discriminator(wave_rppg,fake_rppg)
        #print('fake_rppg:',fake_rppg.size())
        #print('pred_fake',pred_fake.size())
        
        #G_loss = MSE(D-1) + 10*L1(ppg-G(rppg))+10*L1(fft(ppg)-fft(G(rppg)))
        ppg_f = fft(wave_gt,1024,norm='forward')
        fake_rppg_f = fft(fake_rppg,1024,norm='forward')
        ppg_f_norm = abs(ppg_f)[:,:,range(int(1024/2))]
        fake_rppg_f_norm = abs(fake_rppg_f)[:,:,range(int(1024/2))]
        time_loss = criterion_freq(wave_gt,fake_rppg)
        freq_loss = criterion_freq(ppg_f_norm,fake_rppg_f_norm)
        GAN_loss = criterion_GAN(pred_fake,valid)
        G_loss = GAN_loss + 10.0*time_loss + 10.0*freq_loss
        #print(g_loss)
        #batch_g_loss_list.append(G_loss)
        
        G_loss.backward()
        
        #optimizer_G.step()
        #scheduler_G.step(G_loss)
        
        # ---------------------
        #  Train Discriminator
        # ---------------------
        
        optimizer1.zero_grad()
        
        # Loss for real 
        validity_real = discriminator(wave_rppg, wave_gt)
        d_real_loss = criterion_GAN(validity_real, valid_smoothed)
        #d_real_loss.backward()#new
        
        # Loss for fake 
        validity_fake = discriminator(wave_rppg,fake_rppg.detach())
        d_fake_loss = criterion_GAN(validity_fake, fake)
        #d_fake_loss.backward()#new
        #Total D Loss
        D_Loss = d_real_loss + d_fake_loss
        #batch_d_loss_list.append(D_Loss)
        D_Loss.backward()
        optimizer1.step()
        
    if (epoch) % sample_interval == 0:
        print(sample_interval, epoch)
        s_path = os.path.join(opt.save_model_path, 'cpt', opt.study)
        os.makedirs(s_path, exist_ok=True)
        
        state_G = {'epoch': epoch, 'state_dict': model.state_dict(),
                   'optimizer': optimizer.state_dict(),'kwargs': kwargs}
        torch.save(state_G, os.path.join(s_path, 'save_{}.pth'.format((epoch))))
        # state_D = {'epoch': epoch, 'state_dict': discriminator.state_dict(),
                   # 'optimizer': optimizer1.state_dict()}
        # torch.save(state_D, os.path.join(s_path, 'D_{}.pth'.format((epoch))))
    return G_runing_loss, D_runing_loss



def validation(model, val_loader, best_loss, sample_interval):
    model.eval()
    eval_loss = 0
    n = 0
    with torch.no_grad():
        for i, (wave_rppg, wave_gt,exp_ppg_peaks) in enumerate(tqdm(val_loader)):
            batch_size = wave_rppg.shape[0]
            #noise = torch.rand(wave_rppg.shape)
            wave_rppg = Variable(wave_rppg.type(Tensor)).cuda().detach()
            wave_gt = Variable(wave_gt.type(Tensor)).cuda().detach()
            exp_ppg_peaks = Variable(exp_ppg_peaks.type(Tensor)).cuda().detach()
            #noise = Variable(noise.type(Tensor)).cuda().detach()
            
            output = model(wave_rppg)
            if opt.Loss == 'exp':
                loss = criterion(output, wave_gt,exp_ppg_peaks)
            else:    
                loss = criterion(output, wave_gt)
            #scheduler.step(loss)
            eval_loss += (loss.detach().cpu().item() * batch_size)
            n += batch_size

            ###############save model##################

        print('epoch:{} loss:{} pre_best loss:{} '.format(
            epoch, eval_loss, best_loss))
        is_best = eval_loss < best_loss

        if is_best:
            print('save best model')
            s_path = os.path.join(opt.save_model_path, 'cpt', opt.study)
            os.makedirs(s_path, exist_ok=True)
            state = {'epoch': epoch,
                     'state_dict': model.state_dict(),
                     'optimizer': optimizer.state_dict(),
                     'kwargs': kwargs
                     }

            torch.save(state, os.path.join(
                s_path, 'save_{}_best.pth'.format((epoch))))

        else:
            if (epoch) % sample_interval == 0 or epoch == (opt.n_epochs):
                print(sample_interval, epoch)
                s_path = os.path.join(opt.save_model_path, 'cpt', opt.study)
                os.makedirs(s_path, exist_ok=True)
                state = {'epoch': epoch,
                         'state_dict': model.state_dict(),
                         'optimizer': optimizer.state_dict(),
                         'kwargs': kwargs
                         }

                torch.save(state, os.path.join(
                    s_path, 'save_{}.pth'.format((epoch))))

        best_loss = min(eval_loss, best_loss)

    return eval_loss, best_loss

def validation_GAN(model, discriminator, val_loader, sample_interval):
    model.eval()
    eval_G_Loss = 0
    eval_D_Loss = 0
    n = 0
    #with torch.no_grad():
    for i, (wave_rppg, wave_gt,exp_ppg_peaks) in enumerate(tqdm(val_loader)):
        batch_size = wave_rppg.shape[0]
        
        wave_rppg = Variable(wave_rppg.type(Tensor)).cuda()#.detach()
        wave_gt = Variable(wave_gt.type(Tensor)).cuda()#.detach()
        exp_ppg_peaks = Variable(exp_ppg_peaks.type(Tensor)).cuda()#.detach()
        
        valid = Variable(Tensor(batch_size, 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(batch_size, 1).fill_(0.0), requires_grad=False)
        valid_smoothed = torch.add(valid, -0.1)
        output = model(wave_rppg)
        
        if opt.Model == 'WGAN_GP':
            if opt.dis_loss == 'GN':
                validity_real = normalize_gradient(discriminator, wave_gt, wave_rppg)
                validity_fake = normalize_gradient(discriminator, output, wave_rppg)
            else:
                validity_real = discriminator(wave_rppg, wave_gt)
                validity_fake = discriminator(wave_rppg, output)
                gradient_penalty = compute_gradient_penalty(
                    discriminator, wave_gt.data, output.data, wave_rppg)
            #scheduler.step(loss)
            
            if opt.dis_loss == 'hinge':
                D_Loss = torch.mean(nn.ReLU()(1.0 - validity_real)) + torch.mean(nn.ReLU()(1.0 + validity_fake))
                    
            elif opt.dis_loss == 'WGAN_GP':
                #print('gp_loss')
                D_Loss = -torch.mean(validity_real) + torch.mean(validity_fake) + lambda_gp * gradient_penalty
                    
            elif opt.dis_loss == 'GN':
                D_Loss = -torch.mean(validity_real) + torch.mean(validity_fake)
            
            if opt.dis_loss == 'GN':
                pred_fake = normalize_gradient(discriminator, output, wave_rppg)
            else:
                pred_fake = discriminator(wave_rppg, output)
            G_loss = -torch.mean(pred_fake)
        
        else:         
            fake_rppg = model(wave_rppg)
            pred_fake = discriminator(wave_rppg,fake_rppg)
            #print('fake_rppg:',fake_rppg.size())
            #print('pred_fake',pred_fake.size())
            
            #G_loss = MSE(D-1) + 10*L1(ppg-G(rppg))+10*L1(fft(ppg)-fft(G(rppg)))
            ppg_f = fft(wave_gt,1024,norm='forward')
            fake_rppg_f = fft(fake_rppg,1024,norm='forward')
            ppg_f_norm = abs(ppg_f)[:,:,range(int(1024/2))]
            fake_rppg_f_norm = abs(fake_rppg_f)[:,:,range(int(1024/2))]
            time_loss = criterion_freq(wave_gt,fake_rppg)
            freq_loss = criterion_freq(ppg_f_norm,fake_rppg_f_norm)
            GAN_loss = criterion_GAN(pred_fake,valid)
            G_loss = GAN_loss + 10.0*time_loss + 10.0*freq_loss
            
            
            validity_real = discriminator(wave_rppg, wave_gt)
            d_real_loss = criterion_GAN(validity_real, valid_smoothed)
            #d_real_loss.backward()#new
             
            # Loss for fake 
            validity_fake = discriminator(wave_rppg,fake_rppg.detach())
            d_fake_loss = criterion_GAN(validity_fake, fake)
            #d_fake_loss.backward()#new
            #Total D Loss
            D_Loss = d_real_loss + d_fake_loss
        
        eval_G_Loss += (G_loss.detach().cpu().item() * batch_size)
        eval_D_Loss += (D_Loss.detach().cpu().item() * batch_size)
        
        n += batch_size

            ###############save model##################

      
    return eval_G_Loss, eval_D_Loss

def test(model, test_loader):

    pea_rate_r, pea_rate_tr = [], []
    MAE_list, MAE_list_tr = [], []
    RMSE_list, RMSE_list_tr = [], []
    HR_AE_list, HR_wave_gt_list, HR_wave_rppg_list = [], [], []
    # IBI_list = []
    # os.makedirs('Fig/{}'.format(opt.sub), exist_ok=True)
    model.eval()
    with torch.no_grad():
        for i, (wave_rppg, wave_gt,exp_ppg_peaks) in enumerate(test_loader):
            #print('i:',i)
            # for i, (wave_rppg_t,wave_gt_t,txt_rppg,txt_ppg) in enumerate(tqdm(test_loader)):
            #if i < 30:
            #noise = torch.rand(wave_rppg.shape)
            wave_rppg = Variable(wave_rppg.type(Tensor)).cuda().detach()

            wave_gt = Variable(wave_gt.type(Tensor)).cuda().detach()
            #noise = Variable(noise.type(Tensor)).cuda().detach()
            # print(wave_gt.size())
            # wave_gt = wave_gt.type(Tensor).detach()
            #wave_rppg = noise
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
                
                pea_rate_r.append(rate_rppg)
                #print('pea_rate:',pea_rate_r)
                pea_rate_tr.append(rate_tr)
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
        
        #print(HR_AE_list)
        #print(HR_wave_gt_list)
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
        
        # MAE_HR_rppg.append(
            # MAE(np.array(HR_wave_rppg_list), np.array(HR_wave_gt_list)))
        # MAE_HR_AE.append(
            # MAE(np.array(HR_AE_list), np.array(HR_wave_gt_list)))
        
        #pea_hr_rppg,_ = pearsonr(HR_wave_rppg_list,HR_wave_gt_list)
        #pea_hr_AE,_ = pearsonr(HR_AE_list,HR_wave_gt_list)
        
        #pea_HR_rppg.append(pea_hr_rppg)        
        #pea_HR_AE.append(pea_hr_AE)
        
        RMSE_HR_rppg.append(
            RMSE_cal(np.array(HR_wave_rppg_list), np.array(HR_wave_gt_list)))
        RMSE_HR_AE.append(RMSE_cal(np.array(HR_AE_list),
                          np.array(HR_wave_gt_list)))
        
        
        print('HR_lens:',len(RMSE_HR_rppg))
        
        out_l.append(output_)
        
        #peak_rppg.append(peak_wave_rppg)
        #peak_ppg.append(peak_wave_gt)
    return a_pea_r, a_pea_tr, a_MAE_r, a_MAE_tr,a_RMSE_r,a_RMSE_tr, RMSE_HR_rppg, RMSE_HR_AE, out_l, wave_rppg_, wave_gt_, output_,HR_AE_list,HR_wave_rppg_list,HR_wave_gt_list,pea_rppg,pea_ml,MAE_rppg,MAE_ml


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

train_loss, train_loss_G, train_loss_D = [], [], []
eval_loss = []

#############################train model#######################################################
best_loss = 10000
for epoch in range(1, opt.n_epochs+1):
    print('train_loader:', len(train_loader))
    ###################
    # train the model #
    ###################
    if opt.Model == 'cGAN':
        loss_G, loss_D = train_GAN(
            model, discriminator, train_loader, epoch, opt.sample_interval)
        train_loss_G_epoch = loss_G/len(train_loader)
        train_loss_G.append(train_loss_G_epoch)
        train_loss_D_epoch = loss_D/len(train_loader)
        train_loss_D.append(train_loss_D_epoch)
        #e_G_loss, e_D_loss = validation_GAN(model, discriminator, val_loader, opt.sample_interval)
        #scheduler.step(e_G_loss)
        #scheduler1.step(e_D_loss)
        print("Epoch {}: Train G loss: {}, Train D loss: {}".format(epoch,
                                                                    np.mean(
                                                                        train_loss_G_epoch),
                                                                    np.mean(train_loss_D_epoch)))
    
    elif  opt.Model == 'WGAN_GP':
        
        loss_G, loss_D = train_WGAN(
            model, discriminator, train_loader, epoch, opt.sample_interval)
        train_loss_G_epoch = loss_G/len(train_loader)
        train_loss_G.append(train_loss_G_epoch)
        train_loss_D_epoch = loss_D/len(train_loader)
        train_loss_D.append(train_loss_D_epoch)
        #e_G_loss, e_D_loss = validation_GAN(model, discriminator, val_loader, opt.sample_interval)
        #scheduler.step(e_G_loss)
        #scheduler1.step(e_D_loss)
        print("Epoch {}: Train G loss: {}, Train D loss: {}".format(epoch,
                                                                    np.mean(
                                                                        train_loss_G_epoch),
                                                                    np.mean(train_loss_D_epoch)))
        
    
    else:
        loss_tr = train(model, train_loader, epoch)
        train_loss_epoch = loss_tr/len(train_loader)
        train_loss.append(train_loss_epoch)

        l_eval, best_loss = validation(
            model, val_loader, best_loss, opt.sample_interval)
        val_loss_epoch = l_eval/len(val_loader)
        scheduler.step(val_loss_epoch)
        eval_loss.append(val_loss_epoch)

        print("Epoch {}: Train loss: {}, Validation loss: {}".format(epoch,
                                                                     np.mean(
                                                                         train_loss_epoch),
                                                                     np.mean(val_loss_epoch)))
    

    a_pea_r, a_pea_tr, a_MAE_r, a_MAE_tr,a_RMSE_r,a_RMSE_tr, RMSE_HR_rppg, RMSE_HR_AE, out_l, wave_rppg_, wave_gt_, output_,HR_AE_list,HR_wave_rppg_list,HR_wave_gt_list,pea_rppg,pea_ml,MAE_rppg,MAE_ml = test(
                model, test_loader)

    # print(wave_gt_.type())
    os.makedirs('{}/Summary/{}'.format(opt.save_model_path,
                opt.study), exist_ok=True)

    x1 = np.arange(len(wave_gt_))
    plt.subplot(311)
    plt.plot(x1, wave_gt_, 'g', linewidth=2, label='gt')
    plt.legend(loc='upper right')
    plt.subplot(312)
    plt.plot(x1, wave_rppg_, 'g', linewidth=2, label='rppg')
    plt.legend(loc='upper right')
    plt.subplot(313)
    plt.plot(x1, output_, 'b', linewidth=2, label=opt.Model)
    # plt.plot(x[0,(t0-10):].cpu().detach().squeeze().numpy(),output_[0,(t0-1-10):(t0+24-1-10)].cpu().detach().squeeze().numpy(),'b--',linewidth=3) # missing data
    # plt.xlabel("x",fontsize=20)
    plt.legend(loc='upper right')
    plt.savefig('{}/Summary/{}/{}_{}.png'.format(opt.save_model_path,
                opt.study, opt.study, epoch))
    #plt.show()
    plt.close()

if opt.Model != 'cGAN' and opt.Model != 'WGAN_GP':
    os.makedirs('{}/Loss'.format(opt.save_model_path), exist_ok=True)
    plt.figure(figsize=(10, 10))
    plt.plot(train_loss)
    plt.plot(eval_loss)

    y1_min = np.argmin(train_loss)
    y1_max = np.argmax(train_loss)
    show_min = str(train_loss[y1_min])
    show_max = str(train_loss[y1_max])

    plt.plot(y1_min, train_loss[y1_min], 'ko')
    plt.plot(y1_max, train_loss[y1_max], 'ko')

    plt.annotate(show_min, xy=(y1_min, train_loss[y1_min]), xytext=(
        y1_min, train_loss[y1_min]))
    plt.annotate(show_max, xy=(y1_max, train_loss[y1_max]), xytext=(
        y1_max, train_loss[y1_max]))

    plt.legend(['Train Loss', 'Eval Loss'], fontsize=25)
    plt.xlabel("Epoch", fontsize=25)
    plt.ylabel("MSE Loss", fontsize=25)
    plt.savefig('{}/Loss/loss_{}.png'.format(opt.save_model_path, opt.study))
    # plt.show()

    with open('{}/{}/Loss_{}.txt'.format(opt.save_model_path, 'Loss', opt.study), 'w') as file:
        file.write("train val"+"\n")
        for i in range((len(train_loss))):
            file.write(str(train_loss[i]) + " " + str(eval_loss[i])
                       + "\n")


os.makedirs('{}/MAE'.format(opt.save_model_path), exist_ok=True)
x_ = range(len(a_MAE_r))
print('len a_MAE_r:', len(a_MAE_r))
plt.figure(figsize=(10, 10))
plt.plot(x_, a_MAE_r, 'g', linewidth=2, label='MAE_rppg')
plt.plot(x_, a_MAE_tr, 'b', linewidth=2, label='MAE_{}'.format(opt.Model))

y1_min = np.argmin(a_MAE_tr)
y1_max = np.argmax(a_MAE_tr)
show_min = str(a_MAE_tr[y1_min])
show_max = str(a_MAE_tr[y1_max])

plt.plot(y1_min, a_MAE_tr[y1_min], 'ko')
plt.plot(y1_max, a_MAE_tr[y1_max], 'ko')

plt.annotate(show_min, xy=(y1_min, a_MAE_tr[y1_min]), xytext=(
    y1_min, a_MAE_tr[y1_min]))
plt.annotate(show_max, xy=(y1_max, a_MAE_tr[y1_max]), xytext=(
    y1_max, a_MAE_tr[y1_max]))
plt.legend(loc='upper right')
# plt.xticks(x_,epoch_list)

plt.title('MAE for epochs:{}'.format(opt.study))
plt.savefig('{}/MAE/MAE_{}.png'.format(opt.save_model_path, opt.study))
plt.close()

# os.makedirs('MAE', exist_ok=True)
x_ = range(len(a_MAE_r))
print('RMSE_HR_rppg:', RMSE_HR_rppg)
plt.figure(figsize=(10, 10))
plt.plot(x_, RMSE_HR_rppg, 'g', linewidth=2, label='rppg')
plt.plot(x_, RMSE_HR_AE, 'b', linewidth=2, label='{}'.format(opt.Model))

y1_min = np.argmin(RMSE_HR_AE)
y1_max = np.argmax(RMSE_HR_AE)
show_min = str(RMSE_HR_AE[y1_min])
show_max = str(RMSE_HR_AE[y1_max])

plt.plot(y1_min, RMSE_HR_AE[y1_min], 'ko')
plt.plot(y1_max, RMSE_HR_AE[y1_max], 'ko')

plt.annotate(show_min, xy=(y1_min, RMSE_HR_AE[y1_min]), xytext=(
    y1_min, RMSE_HR_AE[y1_min]))
plt.annotate(show_max, xy=(y1_max, RMSE_HR_AE[y1_max]), xytext=(
    y1_max, RMSE_HR_AE[y1_max]))
plt.legend(loc='upper right')
# plt.xticks(x_,epoch_list)

plt.legend(loc='upper right')
# plt.xticks(x_,epoch_list)
plt.title('HR RMSE for epochs:{}'.format(opt.study))
plt.savefig('{}/MAE/HR_RMSE_{}.png'.format(opt.save_model_path, opt.study))
plt.close()

x_ = range(len(a_MAE_r))
plt.figure(figsize=(10, 10))
plt.plot(x_, a_pea_r, 'g', linewidth=2, label='pea_rppg')
plt.plot(x_, a_pea_tr, 'b', linewidth=2, label='pea_{}'.format(opt.Model))

y1_min = np.argmin(a_pea_tr)
y1_max = np.argmax(a_pea_tr)
show_min = str(a_pea_tr[y1_min])
show_max = str(a_pea_tr[y1_max])

plt.plot(y1_min, a_pea_tr[y1_min], 'ko')
plt.plot(y1_max, a_pea_tr[y1_max], 'ko')

plt.annotate(show_min, xy=(y1_min, a_pea_tr[y1_min]), xytext=(
    y1_min, a_pea_tr[y1_min]))
plt.annotate(show_max, xy=(y1_max, a_pea_tr[y1_max]), xytext=(
    y1_max, a_pea_tr[y1_max]))
plt.legend(loc='upper right')
# plt.xticks(x_,epoch_list)
plt.legend(loc='upper right')
# plt.xticks(x_,epoch_list)

plt.title('Pearson rate for epochs:{}'.format(opt.study))
plt.savefig('{}/MAE/pea_{}.png'.format(opt.save_model_path, opt.study))
# plt.show()
plt.close()


with open('{}/MAE/MAE_{}.txt'.format(opt.save_model_path, opt.study), 'w') as file:
    file.write("MAE pearson RMSE_HR_rppg RMSE_HR_Model"+"\n")
    for i in range((len(a_MAE_r))):
        file.write(str(a_MAE_tr[i]) + " " + str(a_pea_tr[i]) + " " + str(RMSE_HR_rppg[i])
                   + " " + str(RMSE_HR_AE[i]) + "\n")

# with open('{}/MAE_1amp_{}.txt'.format('MAE',opt.study),'w') as file:
    # file.write("MAE pearson"+"\n")
    # for i in range((len(a_MAE_r))):
        # file.write(str(float(a_MAE_tr[i])/60) + " " + str(float(a_pea_tr[i])/60) + " " +
        # str(float(a_RMSE_fake[i])/60) + "\n")

out_l = np.array(out_l)
print(out_l.shape)

# print(pulseGAN.shape)
# peak_fake_rppg1_list = np.array(peak_fake_rppg1_list)
# print(wave_gt_[(t0-1):(t0+p_len-1)].shape)
# print(wave_gt_.type())
# print(out_l.shape)
# print(y.shape)
x1 = np.arange(len(wave_gt_))
x2 = np.arange(len(wave_gt_))
# figsize=(10,10)
fig, ax1 = plt.subplots()
# ax1.figure(figsize=(10,10))
# def update(i):
# plt.subplot(311)
ax1.plot(x2, wave_gt_, 'g', linewidth=1, label='PPG')
ax1.plot(x2, wave_rppg_, 'b', linewidth=1, label='RPPG')
line, = ax1.plot(x2, out_l[0, :], 'r', linewidth=1,
                 label='{}'.format(opt.Model))
# p, = ax3.plot([],[],"x")
L = ax1.text(1, 1, '', fontsize='large')
# lines.append(line)
# p, = ax3.plot([],[], "x")
# print(peak_fake_rppg)
ax1.legend(loc='upper right')
ax1.set_title(str(opt.study))


def update(k):
    print(k)
    # if k < 10:
    x = np.arange(len(wave_gt_))
    y = out_l[k, :]
    line.set_data(x, y)
    L.set_text(k)

    return line, L,


def init():
    # ax1.set_xlim(300-t0,300)
    # ax1.set_ylim(0,1.3)
    line.set_data(x1, out_l[0, :])
    L.set_text('')
    # p.set_data(peak_fake_rppg1_list[0],pulseGAN[0][peak_fake_rppg1_list[0]])
    return line, L,


os.makedirs('{}/gif/'.format(opt.save_model_path), exist_ok=True)


ani = animation.FuncAnimation(fig, update, range(
    0, opt.n_epochs), init_func=init, interval=1000, blit=True, repeat=False)

ani.save('{}/gif/{}.gif'.format(opt.save_model_path, opt.study), writer='pillow')
# plt.show()
plt.close()


# plt.plot(x2,wave_gt_,'g',linewidth=1,label = 'PPG')
# plt.plot(x2,y,'b',linewidth=1,label = 'RPPG')
# plt.plot(x2[(t0-1):(t0+p_len-1)],out_l[-1,(t0-1):(t0+p_len-1)],'r',linewidth=1,label = 'Trans')
# plt.legend(loc='upper right')
# plt.savefig('{}_last.png'.format(opt.study))
