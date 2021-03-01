# -*- coding:utf-8 -*-
# @Time       :2021/1/29 11:29
# @Author     :Xing CHEN
# @Site       :
# @File       :sample_flickr.py
# @Software   :PyCharm
# @Dirction   :None


import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
import os
from torch.autograd import Variable
from torchvision import transforms
from tqdm import tqdm

from build_vocab import Vocabulary
from model import EncoderCNN, DecoderLstm
from PIL import Image
import nltk
import json
from pycocotools.coco import COCO
from nltk.translate.bleu_score import sentence_bleu
import jieba
from eval_score import avg_rouge, cider_all


def main(args):

    transform = transforms.Compose([
        transforms.Scale(args.crop_size),
        transforms.CenterCrop(args.crop_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])

    with open(args.vocab_path,'rb') as f:
        vocab = pickle.load(f)
    with open(args.image_caption,'r') as f:
        json_file = json.load(f)

    encoder = EncoderCNN(args.embed_size)
    decoder = DecoderLstm(args.embed_size,args.hidden_size,len(vocab),args.num_layers)
    encoder.load_state_dict(torch.load(args.encoder_path,map_location='cpu'))
    decoder.load_state_dict(torch.load(args.decoder_path,map_location='cpu'))
    encoder.eval()
    decoder.eval()

    cider_ref0_list = []
    cider_ref1_list = []
    cider_ref2_list = []
    cider_ref3_list = []
    cider_ref4_list = []
    cider_hyps_list = []
    image_id_list = []
    eval_data = {}
    avg_bleu_1 = 0
    avg_bleu_2 = 0
    avg_bleu_3 = 0
    avg_bleu_4 = 0
    avg_rouge_1 = 0
    avg_rouge_2 = 0
    avg_rouge_l = 0
    count = 0
    for i in tqdm(range(len(json_file))):
    # for i in range(20):
        count += 1
        captions = json_file[i]['caption']
        filename = json_file[i]['image_id']
        # tokens = nltk.tokenize.word_tokenize(caption.lower())
        # caption = []
        # caption.append('<start>')
        # caption.extend([token for token in tokens])
        # caption.append('<end>')
        # img_id = coco.anns[ids[i]]['image_id']
        path = json_file[i]['image_id']
        image_path = os.path.join(args.image_dir,path)
        image = Image.open(image_path)
        image_tensor = Variable(transform(image).unsqueeze(0))
        state = (Variable(torch.zeros(args.num_layers,1,args.hidden_size)),
                 Variable(torch.zeros(args.num_layers,1,args.hidden_size)))
        if torch.cuda.is_available():
            encoder.cuda()
            decoder.cuda()
            state = [s.cuda() for s in state]
            image_tensor = image_tensor.cuda()
        feature = encoder(image_tensor)
        sampled_ids = decoder.sample(feature,state)
        sampled_ids = sampled_ids.cpu().data.numpy()
        # print(sampled_ids)
        sampled_caption = []
        for word_id in sampled_ids:
            word = vocab.idx2word[word_id]
            sampled_caption.append(word)
            if word == '<end>':
                break

        sampled_caption_remove = sampled_caption
        if "<end>" in sampled_caption:
            sampled_caption_remove.remove('<end>')
        if '<start>' in sampled_caption:
            sampled_caption_remove.remove('<start>')
        captions_cut = []
        for i in range(len(captions)):
            seg_list = jieba.cut(captions[i])  # 默认是精确模式
            res = " ".join(seg_list)
            captions_cut.append(res)
        if len(sampled_caption_remove) <= 0:
            continue
        if len(captions_cut) <= 0:
            continue
        sentence_remove =' '.join(sampled_caption_remove)
        # print("reals:",captions)
        # print("reals_cut",captions_cut)
        # print("samples: ", sentence_remove)
        result = {}
        bleu_1 = sentence_bleu(captions_cut,sentence_remove, weights=[1,0,0,0])
        bleu_2 = sentence_bleu(captions_cut,sentence_remove, weights=[0.5,0.5,0,0])
        bleu_3 = sentence_bleu(captions_cut,sentence_remove, weights=[0.33, 0.33, 0.33,0])
        bleu_4 = sentence_bleu(captions_cut,sentence_remove, weights=[0.25, 0.25, 0.25, 0.25])
        result['bleu_1'] = bleu_1
        result['bleu_2'] = bleu_2
        result['bleu_3'] = bleu_3
        result['bleu_4'] = bleu_4
        rouge_1,rouge_2,rouge_l = avg_rouge(sentence_remove,captions_cut)
        result['rouge_1'] = rouge_1
        result['rouge_2'] = rouge_2
        result['rouge_l'] = rouge_l
        result['captions_cut'] = captions_cut
        result['sample'] = sentence_remove

        avg_bleu_1 += bleu_1
        avg_bleu_2 += bleu_2
        avg_bleu_3 += bleu_3
        avg_bleu_4 += bleu_4
        avg_rouge_1 += rouge_1
        avg_rouge_2 += rouge_2
        avg_rouge_l += rouge_l
        cider_hyps_list.append(sentence_remove)
        cider_ref0_list.append(captions_cut[0])
        cider_ref1_list.append(captions_cut[1])
        cider_ref2_list.append(captions_cut[2])
        cider_ref3_list.append(captions_cut[3])
        cider_ref4_list.append(captions_cut[4])

        image_id_list.append(filename)
        eval_data[filename] = result
    cider_refs_list = []
    cider_refs_list.append(cider_ref0_list)
    cider_refs_list.append(cider_ref1_list)
    cider_refs_list.append(cider_ref2_list)
    cider_refs_list.append(cider_ref3_list)
    cider_refs_list.append(cider_ref4_list)

    score,scores = cider_all(cider_refs_list,cider_hyps_list)
    for idx in range(len(image_id_list)):
        eval_data[image_id_list[idx]]['cider'] = scores[idx]


    avg_bleu_1 /= count
    avg_bleu_2 /= count
    avg_bleu_3 /= count
    avg_bleu_4 /= count
    avg_rouge_1 /= count
    avg_rouge_2 /= count
    avg_rouge_l /= count

    eval_data['total'] = {}
    eval_data['total']['CIDEr'] = score
    eval_data['total']['BLEU1'] = avg_bleu_1
    eval_data['total']['BLEU2'] = avg_bleu_2
    eval_data['total']['BLEU3'] = avg_bleu_3
    eval_data['total']['BLEU4'] = avg_bleu_4
    eval_data['total']['ROUGE1'] = avg_rouge_1
    eval_data['total']['ROUGE2'] = avg_rouge_2
    eval_data['total']['ROUGEL'] = avg_rouge_l
    with open('./val/results'+args.val_index+'.json',"w",encoding='utf-8') as f:
        json.dump(eval_data,f)
    print(eval_data['total'])




        # print(result)
        # plt.imshow(np.asanyarray(image))
        # plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir',type = str,default=r'D:\Data\caption\Flickr8k and Flickr8kCN\Flickr8k_image\Flickr8k_resize')
    parser.add_argument('--image_caption',type = str,default=
    r'D:\Data\caption\Flickr8k and Flickr8kCN\data\flickr8kval.json')
    parser.add_argument('--encoder_path',type = str,default='./models/encoder-10-8200.pkl')
    parser.add_argument('--decoder_path',type=str,default='./models/decoder-10-8200.pkl')
    parser.add_argument('--val_index',type=str,default='10-8200-flckr8kcn')
    parser.add_argument('--vocab_path',type=str,default='./data/vocab.pkl')

    # Model parameters (should be same as paramters in train.py)
    parser.add_argument('--embed_size', type=int , default=256)
    parser.add_argument('--crop_size',type=int,default=224)
    parser.add_argument('--hidden_size',type=int,default=512)
    parser.add_argument('--num_layers',type=int,default=3)
    args = parser.parse_args()


    main(args)
