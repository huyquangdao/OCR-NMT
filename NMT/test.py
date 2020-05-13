from inference.DogCatInference import DogCatInference
from models.transfomers import Encoder, Decoder, Seq2Seq
import argparse
import torch
import pickle
from utils.data_utils import translate_sentence
import time

def parse_arg():

    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_src_file', help='Your training directory', default='data/train/train.en')
    parser.add_argument('--train_des_file', help='Your training directory', default='data/train/train.vi')
    parser.add_argument('--test_src_file', help='Your testing directory', default='data/test/tst2013.en')
    parser.add_argument('--test_des_file', help='Your testing directory', default='data/test/tst2013.vi')


    parser.add_argument('--batch_size',help='Your training batch size',default=16, type = int)
    parser.add_argument('--num_workers', help='number of process', default=2, type = int)
    parser.add_argument('--seed',help='random seed',default=1234, type= int)
    parser.add_argument('--epoch', help='training epochs', default=5, type = int)
    parser.add_argument('--lr',help='learning rate',default=0.001)
    parser.add_argument('--max_lr', help = 'maximum learning rate', default=0.01, type= float)
    parser.add_argument('--val_batch_size', help='Your validation batch size', default=8)
    parser.add_argument('--grad_clip',help='gradient clipping theshold',default=5, type = int)
    parser.add_argument('--grad_accum_step', help='gradient accumalation step', default=1)

    parser.add_argument('--n_classes',help='Number of classes', default=2, type=int)

    parser.add_argument('--pretrained',help='path to pretrained model', default=1, type=bool)
    parser.add_argument('--gpu',help='Number of classes', default=0, type= bool)

    parser.add_argument('--log_dir',help='Log directory path', default='logs', type= str)

    parser.add_argument('--lr_scheduler',help= 'learning rate scheduler', default = 'cyclic')
    parser.add_argument('--n_layers', help ='number of transfomer layer', default = 3, type = int)
    parser.add_argument('--n_heads', help= 'number of attention head', default = 8, type= int)

    parser.add_argument('--pf_dim', help ='position feedforward dimesion', default = 512, type= int)
    parser.add_argument('--hidden_size', help= 'hidden_size', default = 256, type= int)
    parser.add_argument('--drop_out', help = 'drop out prop', default = 0.1, type= float)
    parser.add_argument('--max_seq_length', help='max sequence length', default = 100, type = int)
    
    parser.add_argument('--model', help='vocab size', default = 'weights/nmt_model_best.pth')



    parser.add_argument('--vocab_size', help='vocab size', default = 20000)

    args = parser.parse_args()

    return args


if __name__ == "__main__":

    args = parse_arg()

    with open('weights/SRC.pkl','rb') as f:
        SRC = pickle.load(f)
    
    with open('weights/TRG.pkl','rb') as f:
        TRG = pickle.load(f)

    INPUT_DIM = len(SRC.vocab)
    OUTPUT_DIM = len(TRG.vocab)

    if args.gpu:
        DEVICE = torch.device('cuda:0')
    
    else:
        DEVICE = torch.device('cpu')

    enc = Encoder(INPUT_DIM, 
                    args.hidden_size, 
                    args.n_layers, 
                    args.n_heads, 
                    args.pf_dim, 
                    args.drop_out, 
                    DEVICE,
                    max_length=200)

    dec = Decoder(OUTPUT_DIM, 
                args.hidden_size, 
                args.n_layers, 
                args.n_heads, 
                args.pf_dim, 
                args.drop_out, 
                DEVICE,
                max_length=200)

    SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]
    TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]
                
    model = Seq2Seq(encoder = enc, decoder = dec, src_pad_idx=SRC_PAD_IDX, trg_pad_idx=TRG_PAD_IDX, device= DEVICE)

    model.load_state_dict(torch.load(args.model, map_location=DEVICE))

    s_time = time.time()

    sentence = 'There are three windows in this room.'

    s_time = time.time()

    result, _ = translate_sentence(sentence, SRC, TRG,model,DEVICE, 100)

    print(time.time() - s_time)

    print(result)

