from OCR.detection import TextDetection
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import string
from OCR.pretrained.recognition.utils import CTCLabelConverter,AttnLabelConverter
from OCR.pretrained.recognition.dataset import AlignCollate
from OCR.pretrained.recognition.model import Model
from OCR.data.dataset import MyDataset
from OCR.config.config import DETECTION_CONFIG,RECOGNITION_CONFIG
import cv2
import silx


from NMT.models.transfomers import Encoder, Decoder, Seq2Seq
import argparse
import torch
import pickle
from NMT.utils.data_utils import translate_sentence
import time


from OCR.recognition import TextRecognition


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
    
    parser.add_argument('--model', help='vocab size', default = 'NMT/weights/nmt_model_best.pth')



    parser.add_argument('--vocab_size', help='vocab size', default = 20000)

    args = parser.parse_args()

    return args

if __name__ == "__main__":

    args = parse_arg()
    
    detection = TextDetection(DETECTION_CONFIG.MODEL_PATH, DETECTION_CONFIG.TEXT_THRESHOLD, DETECTION_CONFIG.LOW_TEXT, \
                              DETECTION_CONFIG.LINK_THRESHOLD, DETECTION_CONFIG.CUDA, DETECTION_CONFIG.CANVAS_SIZE, \
                              DETECTION_CONFIG.MAG_RATIO, DETECTION_CONFIG.POLY, DETECTION_CONFIG.SHOW_TIME)

    recognition = TextRecognition(detection,RECOGNITION_CONFIG)
    image_path = 'OCR/test/test6.jpg'
    image = cv2.imread(image_path)

    with open('NMT/weights/SRC.pkl','rb') as f:
        SRC = pickle.load(f)
    
    with open('NMT/weights/TRG.pkl','rb') as f:
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

    text_document = recognition.recognize_text(image)

    lines = text_document.split('\n')

    for line in lines:

        sentence = line

        result, _ = translate_sentence(sentence, SRC, TRG,model,DEVICE, 100)

        print('src {0} -> translate {1}'.format(line,result))

    print(time.time() - s_time)
