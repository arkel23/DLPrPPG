import numpy as np
import math
from torchsummary import summary
import torchsummary
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn.utils import spectral_norm

cuda = True if torch.cuda.is_available() else False
################################################
############LSTM part###########################
################################################


class PPG2ECG_BASELINE_LSTM(nn.Module):
    def __init__(self, **kwargs):
        super(PPG2ECG_BASELINE_LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = 0.1
        self.encoder = nn.LSTM(input_size, self.hidden_size, self.num_layers,
                               dropout=self.dropout, batch_first=True,
                               bidirectional=True)
        output_size = input_size
        self.decoder = nn.LSTM(self.hidden_size*2, output_size,
                               self.num_layers, batch_first=True,
                               bidirectional=True)
        self.linear = nn.Linear(output_size*2, output_size)

    def forward(self, x):
        # print('input:',x.size())
        # print('input_size:',x.size())
        x = x.view(x.size(0), -1, self.input_size)
        # print('view:',x.size())
        encoded_output, (h_n, c_n) = self.encoder(x, None)
        encoded_output = nn.ReLU()(encoded_output)
        # print('encode_size:',encoded_output.size())
        decoded_output, (h_n, c_n) = self.decoder(encoded_output, None)
        # print('decoded_size:',decoded_output.size())
        decoded_output = self.linear(decoded_output)
        # print('linear:',decoded_output.size())
        decoded_output = decoded_output.view(x.size(0), -1, self.input_size)
        # print(decoded_output.size())
        decoded_output = decoded_output.view(decoded_output.size(0), 1, -1)
        # print(decoded_output.size())
        decoded_output = nn.Tanh()(decoded_output)
        return decoded_output


class RNN_model(nn.Module):
    def __init__(self, **kwargs):
        super(RNN_model, self).__init__()
        self.input_size = kwargs['input_size']
        
        self.dropout = kwargs['dropout']
        self.layers = kwargs['num_layers']
        
        self.hidden_size = kwargs['emb_size']
        self.bi = kwargs['bi']
       
        
        # self.embedding = torch.nn.Embedding(2,hidden_size)
        # self.embedding = InEmbedding(input_size,kernel_size,hidden_size,padding_mode)

        self.rnn = nn.LSTM(
            input_size=self.input_size, hidden_size=self.hidden_size,
            num_layers=self.layers, dropout=self.dropout, batch_first=True, bidirectional=self.bi
        )
        #self.relu = nn.ReLU()
        if self.bi:
            self.linear = nn.Linear(self.hidden_size*2, 1)
        else:
            self.linear = nn.Linear(self.hidden_size, 1)
        #self.act = nn.Sigmoid()
        self.act = nn.Tanh()
        
    def forward(self, x):
        # LSTM:(batch, seq, feature)

        # x = self.embedding(x)
        # print(z.size())
        # h_0 = Variable(torch.zeros(
        # self.num_layers, x.size(0), self.hidden_size,device=x.device))#(num_layers * num_directions, batch, proj_size)

        # c_0 = Variable(torch.zeros(
        # self.num_layers, x.size(0), self.hidden_size,device=x.device))
        #print('input:', x.size())
        x = x.permute(0, 2, 1)
        #print('permute:', x.size())
        r_out, (h_out, _) = self.rnn(x)
        #print('r_out:', r_out.size())
        # self.embedding()
        # h_out = h_out.view(-1, self.hidden_size)
        #r_out = self.relu(r_out)
        out = self.linear(r_out).permute(0, 2, 1)
        out = self.act(out)
        # print(ula.size())
        # print(h_out.size())

        return out

############################################################
#####################Gan_part###############################
############################################################


class Encoder_gan(nn.Module):
    def __init__(self, in_size, out_size, kwidth):
        super(Encoder_gan, self).__init__()

        layers = [nn.Conv1d(in_size, out_size, kwidth, 2, padding=kwidth//2),
                  nn.InstanceNorm1d(out_size),
                  nn.PReLU(num_parameters=out_size),
                  # nn.ReLU(inplace=True),
                  # nn.Dropout(p=Dropout)
                  ]
        # layers.append(nn.PReLU(inplace=True))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        z = 0
        if x.size(-1) % 2 != 0:
            z = 1
        return self.model(x), z


class Decoder_gan(nn.Module):
    def __init__(self, in_size, out_size, kwidth, up_mode):
        super(Decoder_gan, self).__init__()
        if up_mode == 'upconv':
            layers = [nn.ConvTranspose1d(in_size, out_size, kwidth, 2, padding=kwidth//2, output_padding=1),
                      nn.InstanceNorm1d(out_size),
                      nn.PReLU(num_parameters=out_size),
                      # nn.ReLU(inplace=True),
                      # nn.Dropout(p=Dropout)
                      ]
        elif up_mode == 'upsample':
            layers = [nn.Upsample(mode='linear', scale_factor=2),
                      nn.Conv1d(in_size, out_size, kwidth,
                                1, padding=kwidth//2),
                      nn.PReLU(num_parameters=out_size),
                      # nn.ReLU(inplace=True),
                      # nn.Dropout(p=Dropout)
                      ]

        # if dropout:
        #    layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input, z):
        x = self.model(x)
        #print('original x:',x.size())
        if z:
            x = x[:, :, :-1]
        #print('reshape x:',x.size())    
        x = torch.cat((x, skip_input), 1)
        # print(x.size())

        return x


class Generator(nn.Module):
    def __init__(self, **kwargs):
        super(Generator, self).__init__()
        
        self.input_size = kwargs['input_size']
        self.n_fc =  kwargs['n_fc']
        self.dropout = kwargs['dropout']
        self.layers = kwargs['num_layers']
        self.kernel_size = kwargs['kwidth']
        self.emb_size = kwargs['emb_size']
        self.num_heads = kwargs['num_heads']
        self.kwidth = kwargs['kwidth']
        self.up_mode = kwargs['up_mode']
        
        self.Encoder_gan_blks = nn.Sequential()
        self.Encoder_gan_blks.add_module(
            "E1", Encoder_gan(self.input_size, self.n_fc, self.kwidth))
        # z_inf = np.zeros(l_side)
        # x,z_inf[0] = self.down1(x)

        in_ker = self.n_fc
        for i in range(0, self.layers-1):
            out_ker = in_ker*2
            self.Encoder_gan_blks.add_module("E{}".format(
                i+2), Encoder_gan(in_ker, out_ker, self.kwidth))
            in_ker = out_ker

        # in_ker = in_ker//2

        self.Decoder_gan_blks = nn.Sequential()
        for i in range(0, self.layers-1):
            out_ker = in_ker//2
            if i == 0:
                self.Decoder_gan_blks.add_module("D{}".format(
                    i+1), Decoder_gan(in_ker, out_ker, self.kwidth, self.up_mode))
            else:
                self.Decoder_gan_blks.add_module("D{}".format(
                    i+1), Decoder_gan(in_ker*2, out_ker, self.kwidth, self.up_mode))
            in_ker = out_ker

        # self.up1 = Decoder_gan(n_fc*32,n_fc*16,kwidth, up_mode)
        # self.up2 = Decoder_gan(n_fc*32, n_fc*8,kwidth, up_mode)
        # self.up3 = Decoder_gan(n_fc*16, n_fc*4,kwidth, up_mode)
        # self.up4 = Decoder_gan(n_fc*8, n_fc*2,kwidth, up_mode)
        # self.up5 = Decoder_gan(n_fc*4, n_fc,kwidth, up_mode)

        if self.up_mode == 'upconv':
            self.final = nn.Sequential(
                # nn.ConvTranspose1d(self.n_fc*2, 1, self.kwidth,
                                  # 2, padding=self.kwidth//2, output_padding=1),
                # nn.InstanceNorm1d(1),
                # nn.Tanh(),
                # nn.Dropout(p=Dropout)
                nn.Upsample(mode='linear', scale_factor=2),
                nn.Conv1d(self.n_fc*2, 1, self.kwidth, 1,padding = self.kwidth//2),
                nn.Tanh(),
            )
        elif self.up_mode == 'upsample':
            self.final = nn.Sequential(nn.Upsample(mode='linear', scale_factor=2),
                                       nn.Conv1d(self.n_fc*2, 1,
                                                 self.kwidth, 1, padding=self.kwidth//2),
                                       nn.Tanh(),
                                       # nn.Dropout(p=Dropout)
                                       )

    def init_weights(self):
        def _init(m):
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                nn.init.xavier_uniform_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)
        self.apply(_init)
        nn.init.constant_(self.model.fc.weight, 0)
        nn.init.constant_(self.model.fc.bias, 0)

    def forward(self, x):
        # U-Net generator with skip connections from Encoder_gan to Decoder_gan
        # z_inf = np.zeros(l_side)
        ii = 0
        z_inf = []
        down_list = []
        up_list = []
        for E in self.Encoder_gan_blks:
            x, z = E(x)
            #print('Encoder_gan{}'.format(ii+1),x.size(),z)
            z_inf.append(z)
            down_list.append(x)
            # print('Encoder_gan:',x.size())
            ii = ii+1
        # print(i)
        k = 0
        for D in self.Decoder_gan_blks:
            ii = ii-1
            # print('K:',k)
            if k == 0:
                x = D(down_list[ii], down_list[ii-1], z_inf[ii])
            else:
                x = D(x, down_list[ii-1], z_inf[ii])
            #print('Decoder_gan:',x.size())
            up_list.append(x)
            # print('u{}'.format(k+1),x.size())
            k = k+1
        x = self.final(x)
        #print('final:',x.size())
        
        return x


class Discriminator(nn.Module):
    def __init__(self, **kwargs):
        super(Discriminator, self).__init__()
        
        self.input_size = kwargs['input_size']
        self.n_fc =  kwargs['n_fc']
        self.dropout = kwargs['dropout']
        self.layers = kwargs['num_layers']
        self.kernel_size = kwargs['kwidth']
        self.emb_size = kwargs['emb_size']
        self.num_heads = kwargs['num_heads']
        self.kwidth = kwargs['kwidth']
        self.up_mode = kwargs['up_mode']
        self.data_len = kwargs['data_len']
        
        self.model = nn.Sequential()
        self.model.add_module("L1", nn.Sequential(nn.Conv1d(2, self.n_fc, self.kwidth, stride=2, padding=self.kwidth//2),
                                                  nn.BatchNorm1d(self.n_fc),
                                                  #nn.InstanceNorm1d(self.n_fc),
                                                  nn.LeakyReLU(0.01,inplace=True),)
                              # nn.Dropout(p=Dropout))
                              )

        in_ker = self.n_fc
        for l_number in range(2, self.layers+1):
            conv = [nn.Conv1d(in_ker, in_ker*2, self.kwidth, stride=2, padding=self.kwidth//2),
                    nn.BatchNorm1d(in_ker*2),
                    #nn.InstanceNorm1d(in_ker*2),
                    nn.LeakyReLU(0.01,inplace=True),
                    # nn.Dropout(p=Dropout)
                    ]
            self.model.add_module("L{}".format(l_number), nn.Sequential(*conv))
            in_ker = in_ker*2

        l_number = l_number+1
        
        #self.final_conv = nn.Conv1d(1, 1, 1, stride=1)
                           
        
        self.model.add_module("L{}".format(l_number), nn.Sequential(nn.Conv1d(in_ker, in_ker*2, self.kwidth, stride=2, padding=self.kwidth//2),
                                                                    nn.LeakyReLU(0.01,inplace=True),)
                              # nn.Dropout(p=Dropout))
                              )

        for i in range(l_number):
            # print(i)
            self.data_len = math.ceil(self.data_len/2)
            #print(self.data_len)
            
        self.linear = nn.Linear(self.data_len*self.n_fc*64, 1)

    def init_weights(self):
        def _init(m):
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                nn.init.xavier_uniform_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)
        self.apply(_init)
        nn.init.constant_(self.model.fc.weight, 0)
        nn.init.constant_(self.model.fc.bias, 0)

    def forward(self, x, gt):
        # Concatenate label embedding and image to produce input
        d_in = torch.cat((x, gt), 1)
        out = self.model(d_in)

        #print(out.size())
        #out = torch.flatten(out, start_dim = 1).unsqueeze(1)
        #print('flat:',out.size())
        #out = self.final_conv(out)
        # print(out1.size()[-1])
        out = torch.flatten(out, start_dim = 1)
        validity = self.linear(out)
        # print(d_in.size())
        # print(out.size())
        # print(out1.size())
        # print(validity.size())
        return validity


class Discriminator_gp(nn.Module):
    def __init__(self, **kwargs):
        super(Discriminator_gp, self).__init__()
        
        self.input_size = kwargs['input_size']
        self.n_fc =  kwargs['n_fc']
        self.dropout = kwargs['dropout']
        self.layers = kwargs['num_layers']
        self.kernel_size = kwargs['kwidth']
        self.emb_size = kwargs['emb_size']
        self.num_heads = kwargs['num_heads']
        self.kwidth = kwargs['kwidth']
        self.up_mode = kwargs['up_mode']
        self.data_len = kwargs['data_len']
        
        self.model = nn.Sequential()
        self.model.add_module("L1", nn.Sequential(nn.Conv1d(2, self.n_fc, self.kwidth, stride=2, padding=self.kwidth//2),
                                                  #nn.BatchNorm1d(self.n_fc),
                                                  #nn.InstanceNorm1d(self.n_fc),
                                                  #nn.LayerNorm(self.n_fc).permute(2,0,1),
                                                  nn.LeakyReLU(0.01,inplace=True)
                                                  )
                              # nn.Dropout(p=Dropout))
                              )

        in_ker = self.n_fc
        for l_number in range(2, self.layers+1):
            conv = [nn.Conv1d(in_ker, in_ker*2, self.kwidth, stride=2, padding=self.kwidth//2),
                    #nn.BatchNorm1d(in_ker*2),
                    #nn.InstanceNorm1d(in_ker*2),
                    #nn.LayerNorm(in_ker*2),
                    nn.LeakyReLU(0.01,inplace=True),
                    # nn.Dropout(p=Dropout)
                    ]
            self.model.add_module("L{}".format(l_number), nn.Sequential(*conv))
            in_ker = in_ker*2

        l_number = l_number+1
        
        #self.final_conv = nn.Conv1d(1, 1, 1, stride=1)
                           
        
        self.model.add_module("L{}".format(l_number), nn.Sequential(nn.Conv1d(in_ker, in_ker*2, self.kwidth, stride=2, padding=self.kwidth//2),
                                                                    nn.LeakyReLU(0.01,inplace=True),)
                              # nn.Dropout(p=Dropout))
                              )

        for i in range(l_number):
            # print(i)
            self.data_len = math.ceil(self.data_len/2)
            #print(self.data_len)
            
        self.linear = nn.Linear(self.data_len*self.n_fc*64, 1)
    

    
    def init_weights(self):
        def _init(m):
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                nn.init.xavier_uniform_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)
        self.apply(_init)
        nn.init.constant_(self.model.fc.weight, 0)
        nn.init.constant_(self.model.fc.bias, 0)
        
    def forward(self, x, gt):
        # Concatenate label embedding and image to produce input
        d_in = torch.cat((x, gt), 1)
        out = self.model(d_in)

        #print(out.size())
        #out = torch.flatten(out, start_dim = 1).unsqueeze(1)
        #print('flat:',out.size())
        #out = self.final_conv(out)
        # print(out1.size()[-1])
        out = torch.flatten(out, start_dim = 1)
        validity = self.linear(out)
        # print(d_in.size())
        # print(out.size())
        # print(out1.size())
        # print(validity.size())
        return validity


class Discriminator_SN(nn.Module):
    def __init__(self, **kwargs):
        super(Discriminator_SN, self).__init__()
        
        self.input_size = kwargs['input_size']
        self.n_fc =  kwargs['n_fc']
        self.dropout = kwargs['dropout']
        self.layers = kwargs['num_layers']
        self.kernel_size = kwargs['kwidth']
        self.emb_size = kwargs['emb_size']
        self.num_heads = kwargs['num_heads']
        self.kwidth = kwargs['kwidth']
        self.up_mode = kwargs['up_mode']
        self.data_len = kwargs['data_len']
        
        self.model = nn.Sequential()
        self.model.add_module("L1", nn.Sequential(spectral_norm(nn.Conv1d(2, self.n_fc, self.kwidth, stride=2, padding=self.kwidth//2)),
                                                  #nn.BatchNorm1d(self.n_fc),
                                                  #nn.InstanceNorm1d(self.n_fc),
                                                  #nn.LayerNorm(self.n_fc).permute(2,0,1),
                                                  nn.LeakyReLU(0.01,inplace=True)
                                                  )
                              # nn.Dropout(p=Dropout))
                              )

        in_ker = self.n_fc
        for l_number in range(2, self.layers+1):
            conv = [spectral_norm(nn.Conv1d(in_ker, in_ker*2, self.kwidth, stride=2, padding=self.kwidth//2)),
                    #nn.BatchNorm1d(in_ker*2),
                    #nn.InstanceNorm1d(in_ker*2),
                    #nn.LayerNorm(in_ker*2),
                    nn.LeakyReLU(0.01,inplace=True),
                    # nn.Dropout(p=Dropout)
                    ]
            self.model.add_module("L{}".format(l_number), nn.Sequential(*conv))
            in_ker = in_ker*2

        l_number = l_number+1
        
        #self.final_conv = nn.Conv1d(1, 1, 1, stride=1)
                           
        
        self.model.add_module("L{}".format(l_number), nn.Sequential(nn.Conv1d(in_ker, in_ker*2, self.kwidth, stride=2, padding=self.kwidth//2),
                                                                    nn.LeakyReLU(0.01,inplace=True),)
                              # nn.Dropout(p=Dropout))
                              )

        for i in range(l_number):
            # print(i)
            self.data_len = math.ceil(self.data_len/2)
            #print(self.data_len)
            
        self.linear = spectral_norm(nn.Linear(self.data_len*self.n_fc*64, 1))
    

    
    def init_weights(self):
        def _init(m):
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                nn.init.xavier_uniform_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)
        self.apply(_init)
        nn.init.constant_(self.model.fc.weight, 0)
        nn.init.constant_(self.model.fc.bias, 0)
        
    def forward(self, x, gt):
        # Concatenate label embedding and image to produce input
        d_in = torch.cat((x, gt), 1)
        out = self.model(d_in)

        #print(out.size())
        #out = torch.flatten(out, start_dim = 1).unsqueeze(1)
        #print('flat:',out.size())
        #out = self.final_conv(out)
        # print(out1.size()[-1])
        out = torch.flatten(out, start_dim = 1)
        validity = self.linear(out)
        # print(d_in.size())
        # print(out.size())
        # print(out1.size())
        # print(validity.size())
        return validity

############################################################################
###########################CNN_autoencoder part#############################
############################################################################


class PPG2ECG(nn.Module):
    def __init__(self, input_size, use_stn=False, use_attention=False):
        super(PPG2ECG, self).__init__()
        self.use_stn = use_stn
        self.use_attention = use_attention
        # build main transformer
        # (batch,channels,lens)
        self.main = nn.Sequential(
            # Encoder_cnn
            nn.Conv1d(1, 32, kernel_size=31, stride=2, padding=15),
            nn.PReLU(32),
            nn.Conv1d(32, 64, 31, 1, 15),
            nn.PReLU(64),
            nn.Conv1d(64, 128, 31, 2, 15),
            nn.PReLU(128),
            # nn.Conv1d(128, 256, 31, 1, 15),
            # nn.PReLU(256),
            # nn.Conv1d(256, 512, 31, 2, 15),
            # nn.PReLU(512),
            # Decoder_cnn
            # nn.ConvTranspose1d(
            # 512, 256, kernel_size=31, stride=2,
            # padding=15, output_padding=1),
            # nn.PReLU(256),
            # nn.ConvTranspose1d(256, 128, 31, 1, 15),
            # nn.PReLU(128),
            nn.ConvTranspose1d(128, 64, 31, 2, 15, 1),
            nn.PReLU(64),
            nn.ConvTranspose1d(64, 32, 31, 1, 15),
            nn.PReLU(32),
            nn.ConvTranspose1d(32, 1, 31, 2, 15, 1),
            nn.Tanh(),
        )

    def init_weights(self):
        def _init(m):
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                nn.init.xavier_uniform_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)
        self.apply(_init)
        nn.init.constant_(self.model.fc.weight, 0)
        nn.init.constant_(self.model.fc.bias, 0)

    def forward(self, x):
        # x1 = x
        # # whether to use stn
        # if self.use_stn:
        # x2 = self.stn(x1)
        # else:
        # x2 = x
        # # whether to use attention network
        # if self.use_attention:
        # attn_weights = F.softmax(self.attn(x2), dim=2)*self.attn_len
        # x3 = x2*attn_weights
        # else:
        # x3 = x2
        # # main transformer
        # x4 = self.main(x3)
        x = self.main(x)
        # return {'output': x4, 'output_stn': x2}
        return x


class Encoder_cnn(nn.Module):
    def __init__(self, in_size, out_size, kwidth, stride):
        super(Encoder_cnn, self).__init__()

        layers = [nn.Conv1d(in_size, out_size, kwidth, stride=stride, padding=kwidth//2),
                  # nn.InstanceNorm1d(out_size),
                  nn.PReLU(out_size),
                  # nn.ReLU(inplace=True),
                  # nn.Dropout(p=Dropout)
                  ]
        # layers.append(nn.PReLU(inplace=True))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        z = 0
        if x.size(-1) % 2 != 0:
            z = 1
        return self.model(x), z


class Decoder_cnn(nn.Module):
    def __init__(self, in_size, out_size, kwidth, up_mode, stride, output_padding):
        super(Decoder_cnn, self).__init__()
        if up_mode == 'upconv':
            layers = [nn.ConvTranspose1d(in_size, out_size, kwidth, stride=stride, padding=kwidth//2, output_padding=output_padding),
                      # nn.InstanceNorm1d(out_size),
                      nn.PReLU(out_size),
                      # nn.ReLU(inplace=True),
                      # nn.Dropout(p=Dropout)
                      ]
        elif up_mode == 'upsample':
            layers = [nn.Upsample(mode='linear', scale_factor=2),
                      # nn.Conv1d(in_size, out_size, kwidth, 1,padding = kwidth//2),
                      nn.PReLU(out_size),
                      # nn.ReLU(inplace=True),
                      # nn.Dropout(p=Dropout)
                      ]

        # if dropout:
        #    layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, z):
        x = self.model(x)
        if z:
            x = x[:, :, :-1]
        # x = torch.cat((x, skip_input),1)
        # print(x.size())

        return x


class CNN_auto(nn.Module):
    def __init__(self, **kwargs):
        super(CNN_auto, self).__init__()
        
        self.input_size = kwargs['input_size']
        self.n_fc =  kwargs['n_fc']
        self.dropout = kwargs['dropout']
        self.layers = kwargs['num_layers']
        self.kernel_size = kwargs['kwidth']
        self.emb_size = kwargs['emb_size']
        self.num_heads = kwargs['num_heads']
        self.kwidth = kwargs['kwidth']
        self.up_mode = kwargs['up_mode']
        
        self.Encoder_cnn_blks = nn.Sequential()
        self.Encoder_cnn_blks.add_module("E1", Encoder_cnn(
            self.input_size, self.n_fc, self.kwidth, stride=2))
        # z_inf = np.zeros(l_side)
        # x,z_inf[0] = self.down1(x)

        in_ker = self.n_fc
        for i in range(0, self.layers-1):

            out_ker = in_ker*2
            # if i%2 ==0:
            # self.Encoder_cnn_blks.add_module("E{}".format(i+2),Encoder_cnn(in_ker,out_ker ,kwidth,stride=1))
            # else:
            self.Encoder_cnn_blks.add_module("E{}".format(
                i+2), Encoder_cnn(in_ker, out_ker, self.kwidth, stride=2))
            in_ker = out_ker

        # in_ker = in_ker//2

        self.Decoder_cnn_blks = nn.Sequential()
        for i in range(0, self.layers-1):
            out_ker = in_ker//2
            # if i==0:
            # if i%2 ==0:
            self.Decoder_cnn_blks.add_module("D{}".format(
                i+1), Decoder_cnn(in_ker, out_ker, self.kwidth, self.up_mode, stride=2, output_padding=1))
            # else:
            # self.Decoder_cnn_blks.add_module("D{}".format(i+1),Decoder_cnn(in_ker,out_ker ,kwidth, up_mode,stride=1,output_padding=0))

            # else:
            # self.Decoder_cnn_blks.add_module("D{}".format(i+1),Decoder_cnn(in_ker*2,out_ker ,kwidth, up_mode))
            in_ker = out_ker

        if self.up_mode == 'upconv':
            self.final = nn.Sequential(
                nn.ConvTranspose1d(self.n_fc, 1, self.kwidth,
                                   2, padding=self.kwidth//2, output_padding=1),
                # nn.InstanceNorm1d(out_channels),
                #nn.Sigmoid(),
                # nn.Dropout(p=Dropout)
                # nn.Upsample(mode='linear', scale_factor=2),
                # nn.Conv1d(n_fc*2, out_channels, kwidth, 1,padding = kwidth//2),
                nn.Tanh(),
            )
        # elif up_mode == 'upsample':
            # self.final = nn.Sequential(nn.Upsample(mode='linear', scale_factor=2),
            # nn.Conv1d(n_fc*2, out_channels, kwidth, 1,padding = kwidth//2),
            # nn.Tanh(),
            # nn.Dropout(p=Dropout)
            #         )

    def init_weights(self):
        def _init(m):
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                nn.init.xavier_uniform_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)
        self.apply(_init)
        nn.init.constant_(self.model.fc.weight, 0)
        nn.init.constant_(self.model.fc.bias, 0)

    def forward(self, x):
        # U-Net generator with skip connections from Encoder_cnn to Decoder_cnn
        # z_inf = np.zeros(l_side)
        ii = 0
        z_inf = []
        down_list = []
        up_list = []
        for E in self.Encoder_cnn_blks:
            x, z = E(x)
            # print(x.size())
            # print('d{}'.format(ii+1),x.size(),z)
            z_inf.append(z)
            down_list.append(x)
            # print('Encoder_cnn:',x.size())
            ii = ii+1
        # print(i)
        k = 0
        for D in self.Decoder_cnn_blks:
            ii = ii-1
            # print('K:',k)
            # if k == 0:
            # x = D(down_list[ii])
            # else:
            # x = D(x,down_list[ii-1],z_inf[ii])
            x = D(x, z_inf[ii])
            # print(x.size())
            # print('Decoder_cnn:',x.size())
            up_list.append(x)
            # print('u{}'.format(k+1),x.size())
            k = k+1
        x = self.final(x)

        return x

##############################################################
###############Transformer part###############################
##############################################################


class Encoderlayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(Encoderlayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        #self.linear1 = nn.Conv1d(in_channels=d_model, out_channels=dim_feedforward,kernel_size=1)
                                   
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        #self.linear2 = nn.Conv1d(in_channels=dim_feedforward, out_channels=d_model,kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = F.gelu

    # def __setstate__(self, state):
        # if 'activation' not in state:
        # state['activation'] = F.relu
        # super(Encoderlayer, self).__setstate__(state)

    def forward(self, src):
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout1(src2)
        #print(src.size())#(len,batch,channel)
        src = self.norm1(src)
        
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))#.permute(1,2,0)))))
        #print(src2.size())
        src = src + self.dropout2(src2)#.permute(2,0,1))
        src = self.norm2(src)
        return src


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model).float()
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # print('position:',position.size())
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # print('div_term:',div_term.size())
        # from index 0,every 2 point
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # print('pe_before:',pe.size())
        pe = pe.unsqueeze(0).permute(1, 0, 2)
        # print('pe_after:',pe.size())
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x:(batch,chnnel,len)
        out = self.pe[:x.size(2), :]  # (len,1,emb_size)
        # print(self.pe[:x.size(0), :].size())
        return out


class InEmbedding(nn.Module):
    def __init__(self, c_in, kernel_size, d_model):  # remove padding mode
        super(InEmbedding, self).__init__()
        # (batch,channels,lens)
        self.Embedding = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=kernel_size, padding=kernel_size//2)
        #self.Embedding = nn.Linear(1,d_model)
                                   
        

    def forward(self, x):
        #print('input:',x.size())#(batch,channel,len)
        #print('Embeding:',self.Embedding(x).size())
        # (batch,channel,len)->(len,batch,chnnel)
        x = self.Embedding(x).permute(2, 0, 1)  # (len,batch,chnnel)
        #x = self.Embedding(x.permute(2, 0, 1))
        #print(x.size())#:(len,batch,chnnel)
        return x

# self.Embedding = nn.Sequential(nn.Conv1d(in_channels=c_in, out_channels=16,
                                   # kernel_size=kernel_size, padding=kernel_size//2),
                                   # nn.PReLU(num_parameters=16),
                                   # nn.Conv1d(in_channels=16, out_channels=32,
                                   # kernel_size=kernel_size, padding=kernel_size//2),
                                   # nn.PReLU(num_parameters=32),
                                   # nn.Conv1d(in_channels=32, out_channels=64,
                                   # kernel_size=kernel_size, padding=kernel_size//2),
                                   # nn.PReLU(num_parameters=64),
                                   # nn.Conv1d(in_channels=64, out_channels=128,
                                   # kernel_size=kernel_size, padding=kernel_size//2),
                                   # nn.PReLU(num_parameters=128),
                                   # nn.Conv1d(in_channels=128, out_channels=256,
                                   # kernel_size=kernel_size, padding=kernel_size//2),
                                   # nn.PReLU(num_parameters=256),
                                   # nn.Conv1d(in_channels=256, out_channels=512,
                                   # kernel_size=kernel_size, padding=kernel_size//2),
                                   # )
                                   



class Attn(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout):
        super(Attn, self).__init__()

        # attn:(lens,batch,channels)
        self.Attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):

        x, _ = self.Attn(x, x, x)
        x = self.norm(x)
        # print('attn_out:',x.permute(1,2,0).size())
        return x


class Trans(nn.Module):
    def __init__(self, **kwargs):
        super(Trans, self).__init__()
                
        self.input_size = kwargs['input_size']
        # self.hidden_size = hidden_size
        self.dropout = kwargs['dropout']
        self.layers = kwargs['num_layers']
        self.kernel_size = kwargs['kwidth']
        self.emb_size = kwargs['emb_size']
        self.num_heads = kwargs['num_heads']
        
        self.embedding = InEmbedding(
            c_in=self.input_size, kernel_size=self.kernel_size, d_model=self.emb_size)
        self.pos_encoder = PositionalEncoding(d_model=self.emb_size,dropout=self.dropout)

        self.encoder = Encoderlayer(d_model=self.emb_size, nhead=self.num_heads,dropout=self.dropout,dim_feedforward=self.emb_size*4)
        
        # self.linear = nn.Sequential(nn.Linear(self.emb_size, self.emb_size),
                                    # nn.GELU(),
                                    # nn.Linear(self.emb_size, 1)
        # )
        self.linear = nn.Linear(self.emb_size, 1)
        #self.act = nn.Sigmoid()
        self.act = nn.Tanh()
   
    @torch.no_grad()
    def init_weights(self):
        def _init(m):
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
            # _trunc_normal(m.weight, std=0.02)  # from .initialization import _trunc_normal
                nn.init.xavier_uniform_(m.weight)
            if hasattr(m, 'bias') and m.bias is not None:
            # nn.init.constant(m.bias, 0)
                nn.init.normal_(m.bias, std=1e-6)
        self.apply(_init)
        #if hasattr(self, 'fc'):
        nn.init.constant_(self.fc.weight, 0)
        nn.init.constant_(self.fc.bias, 0)
        # _trunc_normal(self.positional_embedding.pos_embedding, std=0.02)
        nn.init.normal_(self.pos_encoder, std=0.02)
        #nn.init.constant_(self.class_token, 0)

    def forward(self, x):
        # input=(batch,channels,lens)
        # attn:(lens,batch,channels)
        # cnn:(batch,channels,lens)
        # x = x.permute()
        # self.embedding(x):(len,batch,chnnel)
        #print('emb:',self.embedding(x).size())
        #print('pos:',self.pos_encoder(x).size())
        embed_value = self.embedding(x) + self.pos_encoder(x)

        # print('embed_value:',embed_value.size())#(lens,batch,channels)

        for i in range(self.layers):
            embed_value = self.encoder(embed_value)
            # embed_value = self.down(embed_value)

        #print('emb:',embed_value.size())
        out = self.linear(embed_value).permute(1, 2, 0)
        #print('linear:',self.linear(embed_value).size())
        self.act(out)

        return out

# kwargs = {'input_size':1,'kwidth':7, 'n_fc':16,'data_len':300,'up_mode':'upconv','emb_size':512,'num_layers':6,'num_heads':8,'bi':1,'dropout':0.2}
# # net = Generator(**kwargs)
# # print(net)
# net = Discriminator_gp(**kwargs)
# #summary(net, input_size=(1, 300), batch_size=64, device='cpu')
# x = torch.randn(32, 1, 300)/10
# z = torch.randn(32, 1, 300)
# y = net(x,z)

# print('y:', y.size())
# # #print(y)
