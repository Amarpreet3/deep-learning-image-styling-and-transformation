import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from utils.DatasetAPI import mscoco_dataset
import os, sys
import argparse
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = "true"


mse = tf.keras.losses.MeanSquaredError()
def test(enc_dec, input_img):
    enc_feats = []
    skip_feats = [None]

    x = enc_dec.encoder(0, input_img)
    enc_feats.append(x)
    for l in range(1, 4):
        x = enc_dec.encoder(l, x)
        enc_feats.append(x[0])
        skip_feats.append(x[1])
        x = x[0]

    bfa_feats = enc_feats
    if EXCLUDED > 0:
        bfa_feats = enc_feats[:-EXCLUDED]
    i = 0
    for feature in (bfa_feats):
        x = enc_dec.bfa(i, x, feature)
        i+=1
    x = enc_dec.bfa(i, x, None)

    dec_feats = []
    for l in reversed(range(4)):
        x = enc_dec.decoder(l, x, skip=skip_feats[l])
        dec_feats.append(x)

    loss_rec = mse(input_img, dec_feats[-1])

    loss_feat_rec = mse(enc_feats[-2], dec_feats[0])

    x = enc_dec.encoder(0, x)
    for l in range(1, 4):
        x = enc_dec.encoder(l, x)
        x = x[0]
    loss_percept = mse(enc_feats[-1], x)

    return loss_rec, loss_percept, loss_feat_rec

        
parser = argparse.ArgumentParser()

parser.add_argument("--encoder", '-e', type=str, default='./ckpts/ckpts-relu/encoder',
                    help="the path to the pretrained VGG encoder")
parser.add_argument("--decoder", '-r', type=str, default='./ckpts/ckpts-relu/decoder',
                    help="the path to the pretrained VGG encoder")   
parser.add_argument("--bfa", '-p', type=str, default='./ckpts/ckpts-relu/decoder',
                    help="the path to the pretrained VGG encoder")   
# It is not necessary to use MS-COCO and the image preprocessing specified in the file at './utils/DatasetAPI.py'.
parser.add_argument("--dataset", '-d', type=str, default='./MSCOCO/train2017', 
                    help="the path to the training dataset (MSCOCO)")
    
parser.add_argument("--bfaexclude", '-b', type=int, default=0, 
                    help="last x blocks to exclude from bfa")

args = parser.parse_args()

EXCLUDED=args.bfaexclude

from utils.model_relu import VggDecoder, VggEncoder, BFA
    
class VggEncDec(tf.keras.Model):
    def __init__(self, enc_path, dec_path):
        super(VggEncDec, self).__init__()
        self.encoder = VggEncoder()
        self.decoder = VggDecoder()
        self.encoder.load_weights(enc_path)
        self.decoder.load_weights(dec_path)
        self.bfa = BFA(EXCLUDED) 
        self.bfa.load_weights(args.bfa)       


enc_dec = VggEncDec(args.encoder, args.decoder)
ds = mscoco_dataset(args.dataset)

num_items = len(list(ds))
total_loss = 0
total_loss_rec, total_loss_percept, total_loss_feat_rec =0,0,0
for i, imgs in enumerate(ds):
    loss = test(enc_dec, imgs) 
    _loss = sum(loss)
    total_loss += _loss * (1/num_items)
    loss_rec, loss_percept, loss_feat_rec = loss
    total_loss_rec += loss_rec * (1/num_items)
    total_loss_percept += loss_percept * (1/num_items)
    total_loss_feat_rec += loss_feat_rec * (1/num_items)

    if (i+1) % 10 == 0:
        to_show = f"iter: {i+1}, loss: {', '.join(map(lambda x: str(x.numpy()), loss))}"
        print(to_show)
print("avg total loss: ", total_loss)
print("avg total loss_rec: ", total_loss_rec)
print("avg total loss_percept: ", total_loss_percept)
print("avg total loss_feat_rec: ", total_loss_feat_rec)

        

        

        
            