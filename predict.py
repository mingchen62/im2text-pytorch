#!/usr/bin/env python

from __future__ import division, unicode_literals
import os
import sys
import argparse
import math
import codecs
import torch

import time

from itertools import count

import onmt.io
import onmt.translate
import onmt
import onmt.ModelConstructor
import onmt.modules
import opts

import image_utils
sys.path.append(os.getcwd()+'/scgInklib-0.1.0')
from net.wyun.mer.ink.scgimage import ScgImage

default_buckets ='[[240,100], [320,80], [400,80],[400,100], [480,80], [480,100], [560,80], [560,100], [640,80],[640,100],\
 [720,80], [720,100], [720,120], [720, 200], [800,100],[800,320], [1000,200]]'

parser = argparse.ArgumentParser(
    description='translate.py',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
opts.add_md_help_argument(parser)
opts.translate_opts(parser)

opt = parser.parse_args()

print "opt", opt

def predict( src, src_dir):
    dummy_parser = argparse.ArgumentParser(description='train.py')
    opts.model_opts(dummy_parser)
    dummy_opt = dummy_parser.parse_known_args([])[0]

    print "dummy_opt", dummy_opt
    print "dummy_opt.__dict__", dummy_opt.__dict__

    opt.cuda = opt.gpu > -1
    if opt.cuda:
        torch.cuda.set_device(opt.gpu)

    # Load the model.
    fields, model, model_opt = \
        onmt.ModelConstructor.load_test_model(opt, dummy_opt.__dict__)

    # File to write sentences to.
    out_file = codecs.open(opt.output, 'w', 'utf-8')
    # Test data
    print "src=", src
    print "src_dir=", src_dir
    data = onmt.io.build_dataset(fields, opt.data_type,
                                 src, None,
                                 src_dir=src_dir,
                                 sample_rate=opt.sample_rate,
                                 window_size=opt.window_size,
                                 window_stride=opt.window_stride,
                                 window=opt.window,
                                 use_filter_pred=False)

    # Sort batch by decreasing lengths of sentence required by pytorch.
    # sort=False means "Use dataset's sortkey instead of iterator's".
    data_iter = onmt.io.OrderedIterator(
        dataset=data, device=opt.gpu,
        batch_size=opt.batch_size, train=False, sort=False,
        sort_within_batch=True, shuffle=False)

    start_t =time.time()
    # Translator
    scorer = onmt.translate.GNMTGlobalScorer(opt.alpha, opt.beta)
    translator = onmt.translate.Translator(model, fields,
                                           beam_size=opt.beam_size,
                                           n_best=opt.n_best,
                                           global_scorer=scorer,
                                           max_length=opt.max_length,
                                           copy_attn=model_opt.copy_attn,
                                           cuda=opt.cuda,
                                           beam_trace=opt.dump_beam != "",
                                           min_length=opt.min_length)
    builder = onmt.translate.TranslationBuilder(
        data, translator.fields,
        opt.n_best, opt.replace_unk, opt.tgt)

    cnt=0
    for batch in data_iter:
        batch_data = translator.translate_batch(batch, data)
        translations = builder.from_batch(batch_data)

        for trans in translations:
            cnt+=1
            n_best_preds = [" ".join(pred)
                            for pred in trans.pred_sents[:opt.n_best]]
            out_file.write('\n'.join(n_best_preds))
            out_file.write('\n')
            out_file.flush()

    now_t =time.time()
    print "count ",cnt, now_t -start_t
    if opt.dump_beam:
        import json
        json.dump(translator.beam_accum,
                  codecs.open(opt.dump_beam, 'w', 'utf-8'))
    return n_best_preds


import image_utils
sys.path.append(os.getcwd()+'/scgInklib-0.1.0')
from net.wyun.mer.ink.scgimage import ScgImage

default_buckets ='[[240,100], [320,80], [400,80],[400,100], [480,80], [480,100], [560,80], [560,100], [640,80],[640,100],\
 [720,80], [720,100], [720,120], [720, 200], [800,100],[800,320], [1000,200]]'
outdir='temp'

def preprocess(l):
    filename, postfix, output_filename, crop_blank_default_size, pad_size, buckets, downsample_ratio = l
    postfix_length = len(postfix)
    status = image_utils.crop_image(filename, output_filename, crop_blank_default_size)
    if not status:
        print ('%s is blank, crop a white image of default size!'%filename)
    status = image_utils.pad_image(output_filename, output_filename, pad_size, buckets)
    if not status:
        print ('%s (after cropping and padding) is larger than the largest provided bucket size, left unchanged!'%filename)
        os.remove(output_filename)
        return

    status = image_utils.downsample_image(output_filename, output_filename, downsample_ratio)


def process_scgink(l,request_id):
    try:
        with open(l, 'r') as f:
            scgink_str=''
            line = f.readline()
            while line:
                scgink_str += line
                line = f.readline()
        f.close()
    except requests.exceptions.RequestException as e:  # all exception
        return "ERROR"

    try:
        scgink_data = ScgImage(scgink_str, request_id)
        # empty traces due to scgink data
        if not scgink_data.traces:
            return "ERROR"
        img_file_path = outdir+'/'+request_id+'_input.png'
        #convert to png format
        scgink_data.save_image(img_file_path)

        #preprocess image
        filename, postfix, processed_img, = img_file_path, '.png', outdir+'/'+request_id+'_preprocessed.png',
        crop_blank_default_size, pad_size, buckets, downsample_ratio = [600,60], (8,8,8,8), default_buckets, 2

        l = (filename, postfix, processed_img, crop_blank_default_size, pad_size, buckets, downsample_ratio)
        preprocess(l)
    except :
        return "ERROR"

    return "SUCC"


if __name__ == "__main__":

    scgink_file =outdir+'/1.scgink'
    status= process_scgink(scgink_file,"1")
    print "status", status
    _, basename = os.path.split(scgink_file)
    stem, ext = os.path.splitext(basename)
    img_file_name=stem+'_preprocessed.png'
    
    os.system('echo '+ img_file_name +'>>temp/test.txt');
    res=predict('temp/test.txt','temp')
    print res
