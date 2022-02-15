train.py

'--rppg', type=str, default='pos',        help='rppg Algorithm'
'--epoch', type=int, default=0,
       help='epoch to start training from'
'--train_epoch', type=int, default=0,
       help='which pretrained epoch to load'
'--n_epochs', type=int, default=50,
       help='number of epochs of training'
'--Model', type=str, default='LSTM',
       help='which model to train:CNN_auto,LSTM,GAN,Transformer'
'--GAN', type=str, default='CGAN',
       help='CGAN,WGAN_GP'
'--kwidth', type=int, default=7,
       help='number of kernel width'
'--n_fc', type=int, default=16,
       help='number of start kernels'
'--data_len', type=int, default=300,
       help='number of data points for one sample'
'--data_path', type=str,
default='/edahome/pcslab/pcs04/RPPG/PulseGan/Datasets',        help='path to data'
'--save_model_path', type=str,
default='/edahome/pcslab/pcs04/RPPG/Transformer/CNN/',        help='path to Save model'
'--target_wave', type=str, default='saw',
       help='generator signal:sine,cos,square,saw'
'--gt', type=str, default='sine',        help='ground_truth'
'--up_mode', type=str,
default='upconv',        help='up mode for cnn'
'--batch_size', type=int, default=64,
       help='size of the batches'
'--input_size', type=int, default=1,
       help='input_size for Multi_attn'
'--emb_size', type=int, default=512,
       help='emb_size for Multi_attn'
'--num_layers', type=int, default=3,
       help='num_layers for Transformer/LSTM'
'--num_heads', type=int, default=8,
       help='num_heads for Multi_attn'
"--lr", type=float, default=0.0001,
       help="adam: learning rate"
"--b1", type=float, default=0.5,
       help="adam: decay of first order momentum of gradient"
"--b2", type=float, default=0.999,
       help="adam: decay of first order momentum of gradient"
'--amp', type=int, default=1,        help='parameter of amp'
'--dropout', type=float, default=0.2,
       help='dropout every layers for Multi_attn'
"--SGD", type=bool, default=False,
       help="Use SGD for optimizer"
"--bi", type=bool, default=False,
       help="Bidirectional for LSTM"                    
t('--alpha', type=int, default=10,        help='parameter of time loss'
"--Dis", type=str, default='normal',
       help="use WGAN_GP,SN,normal for D"
"--dis_loss", type=str, default='hinge',
       help="loss for D:WGAN_GP or hinge"
'--beta', type=float, default=0.5,        help='parameter exp_loss'
t('--decay_epoch', type=int, default=50,        help='epoch from which to start lr decay'
'--n_cpu', type=int, default=8,
       help='number of cpu threads to use during batch generation'
'--sample_interval', type=int, default=1,
       help='interval between sampling of images from generators'
'--seed', type=int, default=0,        help='seed for random'
"--n_critic", type=int, default=5,
       help="number of training steps for discriminator per iter"
"--clip_value", type=float, default=0.01,
       help="lower and upper clip value for disc. weights"
'--padding_mode', type=str, default='zeros',
       help='zeros, reflect, replicate or circular'
'--Loss', type=str, default='L1',        help='L1 or L2'
'--study', type=str, default='name',        help='study name'