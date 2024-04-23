
import numpy as np
import tensorflow as tf
from utils.model_relu import VggDecoder, VggEncoder, BFA
import matplotlib.pyplot as plt
import os, time
from PIL import Image
import cv2
from tqdm import tqdm
from utils.photo_gif import GIFSmoothing
os.environ["CUDA_VISIBLE_DEVICES"]="1"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = "true"

class VggEncDec(tf.keras.Model):
    def __init__(self):
        super(VggEncDec, self).__init__()
        self.encoder = VggEncoder()
        self.decoder = VggDecoder()
        self.bfa = BFA(1)
        self.encoder.load_weights('ckpts/ckpts-relu/encoder')
#         self.decoder.load_weights('ckpts/ckpts-relu/decoder')
        self.decoder.load_weights('final_add_3_decoder/decoder')
        self.bfa.load_weights('final_add_3_decoder/bfa')

    def call(self):
        return

enc_dec = VggEncDec()
p_pro = GIFSmoothing(r=50, eps=(0.02 * 255) ** 2)

def load_img(file):
    img = np.asarray(Image.open(file), dtype=np.float32)
    img = np.expand_dims(cv2.resize(img, (img.shape[1] // 8 * 8, img.shape[0] // 8 * 8)), axis=0) / 255
    return img

def inv_sqrt_cov(cov, inverse=False):
    s, u, _ = tf.linalg.svd(cov + tf.eye(cov.shape[-1]))
    n_s = tf.reduce_sum(tf.cast(tf.greater(s, 1e-5), tf.int32))
    s = tf.sqrt(s[:,:n_s])
    if inverse:
        s = 1 / s
    d = tf.linalg.diag(s)
    u = u[:,:,:n_s]
    return tf.matmul(u, tf.matmul(d, u, adjoint_b=True))

def stylize_core(c_feat, s_feat, opt='zca'):
    n_batch, cont_h, cont_w, n_channel = c_feat.shape
    _c_feat = tf.reshape(tf.transpose(c_feat, [0, 3, 1, 2]), [n_batch, n_channel, -1])
    if opt == 'zca':
        c_feat = stylize_zca(_c_feat, s_feat)
    elif opt == 'ot':
        c_feat = stylize_ot(_c_feat, s_feat)
    elif opt == 'adain':
        c_feat = stylize_adain(_c_feat, s_feat)

    c_feat = tf.transpose(tf.reshape(c_feat, [n_batch, n_channel, cont_h, cont_w]), [0, 2, 3, 1])
    return c_feat

def stylize_adain(c_feat, s_feat):
    m_c = tf.reduce_mean(c_feat, axis=-1, keepdims=True)
    m_s = tf.reduce_mean(s_feat, axis=-1, keepdims=True)
    c_feat = c_feat - m_c
    s_feat = s_feat - m_s
    s_c = tf.sqrt(tf.reduce_mean(c_feat * c_feat, axis=-1, keepdims=True) + 1e-8)
    s_s = tf.sqrt(tf.reduce_mean(s_feat * s_feat, axis=-1, keepdims=True) + 1e-8)
    white_c_feat = c_feat / s_c
    feat = white_c_feat * s_s + m_s
    return feat

def stylize_zca(c_feat, s_feat):
    m_c = tf.reduce_mean(c_feat, axis=-1, keepdims=True)
    m_s = tf.reduce_mean(s_feat, axis=-1, keepdims=True)
    c_feat = c_feat - m_c
    s_feat = s_feat - m_s
    c_cov = tf.matmul(c_feat, c_feat, transpose_b=True) / c_feat.shape[-1]
    s_cov = tf.matmul(s_feat, s_feat, transpose_b=True) / s_feat.shape[-1]
    inv_sqrt_c_cov = inv_sqrt_cov(c_cov, True)
    opt = tf.matmul(inv_sqrt_cov(s_cov), inv_sqrt_c_cov)
    feat = tf.matmul(opt, c_feat) + m_s
    return feat

def stylize_ot(c_feat, s_feat):
    m_c = tf.reduce_mean(c_feat, axis=-1, keepdims=True)
    m_s = tf.reduce_mean(s_feat, axis=-1, keepdims=True)
    c_feat = c_feat - m_c
    s_feat = s_feat - m_s
    c_cov = tf.matmul(c_feat, c_feat, transpose_b=True) / c_feat.shape[-1]
    s_cov = tf.matmul(s_feat, s_feat, transpose_b=True) / s_feat.shape[-1]
    sqrt_c_cov = inv_sqrt_cov(c_cov)
    inv_sqrt_c_cov = inv_sqrt_cov(c_cov, True)
    opt = inv_sqrt_cov(tf.matmul(sqrt_c_cov, tf.matmul(s_cov, sqrt_c_cov)))
    opt = tf.matmul(inv_sqrt_c_cov, tf.matmul(opt, inv_sqrt_c_cov))
    feat = tf.matmul(opt, c_feat) + m_s
    return feat

for n in tqdm(range(2)):
    cont_img = load_img(f'figures/content/{n}.jpg')
    style_img = load_img(f'figures/style/{n}.jpg')

    opt = 'zca'

    #getting the output of Vgg 19 for style img
    x1 = enc_dec.encoder(0, style_img)
    x2 = enc_dec.encoder(1, x1)
    x3 = enc_dec.encoder(2, x2[0])
    x4 = enc_dec.encoder(3, x3[0])

    #getting the output of Vgg 19 for content img
    y1 = enc_dec.encoder(0, cont_img)
    y2 = enc_dec.encoder(1, y1)
    y3 = enc_dec.encoder(2, y2[0])
    y4 = enc_dec.encoder(3, y3[0])

    bfax = x4
    #BFA
    bfax = enc_dec.bfa(0, bfax[0], x1)
    bfax = enc_dec.bfa(1, bfax, x2[0])
    bfax = enc_dec.bfa(2, bfax, x3[0])
    bfax = enc_dec.bfa(3,bfax, None)

    bfay = y4
    #BFA
    bfay = enc_dec.bfa(0, bfay[0], y1)
    bfay = enc_dec.bfa(1, bfay, y2[0])
    bfay = enc_dec.bfa(2, bfay, y3[0])
    bfay = enc_dec.bfa(3,bfay, None)

    sfeat = tf.reshape(tf.transpose(bfax, [0, 3, 1, 2]), [bfax.shape[0], bfax.shape[-1], -1])
    x = stylize_core(bfay, sfeat, opt=opt)
    x = enc_dec.decoder(3, x, skip=y4[1])

    sfeat = tf.reshape(tf.transpose(x3[0], [0, 3, 1, 2]), [x3[0].shape[0], x3[0].shape[-1], -1])
    x = stylize_core(x, sfeat, opt=opt)
    x = enc_dec.decoder(2, x, skip=y3[1])

    sfeat = tf.reshape(tf.transpose(x2[0], [0, 3, 1, 2]), [x2[0].shape[0], x2[0].shape[-1], -1])
    x = stylize_core(x, sfeat, opt=opt)
    x = enc_dec.decoder(1, x, skip=y2[1])

    sfeat = tf.reshape(tf.transpose(x1, [0, 3, 1, 2]), [x1.shape[0], x1.shape[-1], -1])
    x = stylize_core(x, sfeat, opt=opt)
    x = tf.clip_by_value(enc_dec.decoder(0, x, skip=None), 0, 1)

    if not os.path.exists('results/relu'):
        os.makedirs('results/relu')
    p_pro.process(x[0], f'figures/content/{n}.jpg').save(f'results/relu/{n}.jpg')



