# -*- coding:utf-8 -*-
# @Time       :2021/1/2 14:14
# @Author     :Xing CHEN
# @Site       :
# @File       :eval_path.py
# @Software   :PyCharm
# @Dirction   :None



from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import cPickle

import models
from eval.dataloaderraw import *
from eval import eval_utils
import argparse
import data.misc.utils as utils
import torch



def eval_path():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='data/log_dense_box_bn/model-best.pth')
    parser.add_argument('--cnn_model', type=str,  default='resnet101')
    parser.add_argument('--infos_path', type=str, default='data/log_dense_box_bn/infos_dense_box_bn-best.pkl')
    parser.add_argument('--image_folder', type=str, default='static/image_input/images')

    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_images', type=int, default=-1)
    parser.add_argument('--language_eval', type=int, default=0)
    parser.add_argument('--dump_images', type=int, default=0)
    parser.add_argument('--dump_json', type=int, default=1)
    parser.add_argument('--dump_path', type=int, default=1)
    parser.add_argument('--sample_max', type=int, default=1)
    parser.add_argument('--max_ppl', type=int, default=0)
    parser.add_argument('--beam_size', type=int, default=5)
    parser.add_argument('--group_size', type=int, default=1)
    parser.add_argument('--diversity_lambda', type=float, default=0.5)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--decoding_constraint', type=int, default=0)
    parser.add_argument('--image_root', type=str, default='')
    parser.add_argument('--input_fc_dir', type=str, default='')
    parser.add_argument('--input_att_dir', type=str, default='')
    parser.add_argument('--input_box_dir', type=str, default='')
    parser.add_argument('--input_label_h5', type=str, default='')
    parser.add_argument('--input_json', type=str, default='')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--coco_json', type=str, default='')

    parser.add_argument('--id', type=str, default='')
    parser.add_argument('--verbose_beam', type=int, default=1)
    parser.add_argument('--verbose_loss', type=int, default=0)

    opt = parser.parse_args()

    with open(opt.infos_path,'rb') as f:
        infos = cPickle.load(f)
    if len(opt.input_fc_dir) == 0:
        opt.input_fc_dir = infos['opt'].input_fc_dir
        opt.input_att_dir = infos['opt'].input_att_dir
        # json 修改
        opt.input_box_dir = infos['opt'].input_box_dir
        opt.input_label_h5 = infos['opt'].input_label_h5

    if len(opt.input_json) == 0:
        opt.input_json = infos['opt'].input_json
    if opt.batch_size == 0:
        opt.batch_size = infos['opt'].batch_size
    if len(opt.id) == 0:
        opt.id = infos['opt'].id
    ignore = ["id", "batch_size", "beam_size", "start_from", "language_eval"]

    print(infos['opt'])
    for k in vars(infos['opt']).keys():
        if k not in ignore:
            if k in vars(opt):
                assert vars(opt)[k] == vars(infos['opt'])[k], k + ' option not consistent'
            else:
                vars(opt).update({k: vars(infos['opt'])[k]}) # copy over options from model



    model = models.setup(opt)
    model.load_state_dict(torch.load(opt.model))
    model.cuda()
    model.eval()
    crit = utils.LanguageModelCriterion()
    loader = DataLoaderRaw({'folder_path': opt.image_folder,
                            'coco_json': opt.coco_json,
                            'batch_size': opt.batch_size,
                            'cnn_model': opt.cnn_model})
    loader.ix_to_word = infos['vocab']

    loss, split_predictions = eval_utils.eval_split(model, crit, loader, vars(opt))

    print('loss: ', loss)

    if opt.dump_json == 1:
        # dump the json
        json.dump(split_predictions, open('static/image_input/results/results.json', 'w'))

if __name__ == '__main__':
    eval_path()