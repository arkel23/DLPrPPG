Train models

python train.py --n_epochs 50 --Model LSTM --GAN WGAN_GP --kwidth 7 --n_fc 16 --data_path path_to_dataset --save_model_path path_to_save_model --emb_size 512 --num_layers 3 --num_heads 8 --lr 0.0001 --Dis WFAN_GP --dis_loss WGAN_GP --sample_interval 1 --Loss exp --study experiment_name

For the important paramaters

--n_epochs: The number of training epochs
--Model: The training networks, including CNN_auto,LSTM,GAN,Transformer
--GAN: Choose the GAN model if the --Model is GAN, including CGAN,WGAN_GP
--kwidth: The number of kernel width for conv1d
--n_fc: The number of start kernels for conv1d
--data_path: The path to /Datasets
--emb_size: The size of emb_size for Multi_attn and hidden_layer for LSTM
--lr: Learning rate
--Dis: The network structure for discriminator in GAN, including WGAN_GP,SN,normal
--dis_loss: The loss function for discriminator, WGAN_GP or hinge
--Loss: The loss function for CNN_auto, LSTM and transformer, including L1, L2, exp

Test models

python test.py --Model_file path_to_model_input --study experiment_name


Information in model_input.txt

model_name(CNN-AE/LSTM/GAN/Transformer)
The path to the .pth file
model_name

For example:
CNN-AE
/edahome/pcslab/pcs04/RPPG/rPPG/Best/CNN_auto/pos_CNN_auto_lr001_n16_b16_k7_layers3_d1/save_10_best.pth
CNN-AE
LSTM
/edahome/pcslab/pcs04/RPPG/rPPG/Best/LSTM/L2/save_17_best.pth
LSTM
GAN
/edahome/pcslab/pcs04/RPPG/rPPG/Best/GAN/best/pos_SN_lr0001_n16_b4_k7_layers6_d1/save_42.pth
GAN
Transformer
/edahome/pcslab/pcs04/RPPG/rPPG/Best/pos_Transformer_k31_emb128_head4_layers6_n16_b32_exp03/save_36_best.pth
Transformer


