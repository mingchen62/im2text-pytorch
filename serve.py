#!/usr/bin/env python

from __future__ import division, unicode_literals
import os
import uuid
import sys
import socket
import requests
import json
from scipy.misc import imread
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
outdir='temp'
debug= True

url = 'http://swissknife:8089/latex_to_asciimath'
payload = {'id':'','asciimath':'', 'mathml':'', 'latex':''}
headers = {'content-type': 'application/json'}

from flask import Flask
app = Flask(__name__)

def get_model_api():
    """Returns lambda function for api"""

    # initialize model once and for all

    # initialize config for translate 
    parser = argparse.ArgumentParser( description='translate.py', formatter_class=argparse.ArgumentDefaultsHelpFormatter) 
    opts.add_md_help_argument(parser)
    opts.translate_opts(parser)
    opt = parser.parse_args()

    # initialize config for model 
    dummy_parser = argparse.ArgumentParser(description='train.py')
    opts.model_opts(dummy_parser)
    dummy_opt = dummy_parser.parse_known_args([])[0]
    opt.cuda = opt.gpu > -1
    if opt.cuda:
        torch.cuda.set_device(opt.gpu)

    # Load the model.
    fields, model, model_opt = \
        onmt.ModelConstructor.load_test_model(opt, dummy_opt.__dict__)
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

    # File to write sentences to.
    out_file = codecs.open(opt.output, 'w', 'utf-8')

    def model_api( input_data):
        """
        Args:
            input_data: submitted to the API, json string

        Returns:
            output_data: after some transformation, to be
                returned to the API

        """


        # process input

        res={}
        request_id=str(uuid.uuid4())
        scgink = input_data["scg_ink"]
        try:
            scgink_data = ScgImage(scgink, request_id)
        except:
            res['status']='error'
            res['info']='bad scgink data'
            return res
        # empty traces due to scgink data
        if not scgink_data.traces:
            res['info']="wrong scgink data"
            res['status']='error'
            return res

        start_t =time.time()

        img_file_path = outdir+'/'+request_id+'_input.png'
        #convert to png format
        scgink_data.save_image(img_file_path)

        #preprocess image
        filename, postfix, processed_img = img_file_path, '.png', outdir+'/'+request_id+'_preprocessed.png'
        crop_blank_default_size, pad_size, buckets, downsample_ratio = [600,60], (8,8,8,8), default_buckets, 2

        l = (filename, postfix, processed_img, crop_blank_default_size, pad_size, buckets, downsample_ratio)
        preprocess(l)

        # construct data
        os.system('echo '+ request_id+'_preprocessed.png ' +'>temp/test.txt');
        src=  'temp/test.txt'
        src_dir='temp'
        #print "src=", src
        #print "src_dir=", src_dir
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

        # Inference
        builder = onmt.translate.TranslationBuilder( data, translator.fields, opt.n_best, opt.replace_unk, opt.tgt)

        cnt=0
        for batch in data_iter:
            batch_data = translator.translate_batch(batch, data)
            translations = builder.from_batch(batch_data)

            for trans in translations:
                cnt+=1
                n_best_preds = [" ".join(pred)
                            for pred in trans.pred_sents[:opt.n_best]]

        if debug:
            now_t =time.time()
            print "time spent ", now_t -start_t

        # process the output
        n_best_latex=[]
        for pred in n_best_preds:
            n_best_latex.append(detokenizer(pred))

        n_best_ascii=[]
        for pred in n_best_latex:
            n_best_ascii.append(latex_asciimath(pred))

        # return the output for the api
        res['status']="succuss"
        res['info']=''
        res['mathml']=''
        res['latex']=n_best_latex[0]
        res['asciimath']=n_best_ascii[0]
        res['n_best_latex']=n_best_latex
        res['n_best_ascii']=n_best_ascii
        app.logger.debug(request_id+"\t"+n_best_latex[0]+"\n");

        return res
    return model_api

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

def detokenizer(s):
    s=s.replace("\left{","{")
    s=s.replace("\left\(","\(")
    s=s.replace("\left[","[")
    s=s.replace("\right}","}")
    s=s.replace("\right\)","\)")
    s=s.replace("\right]","]")
    s=s.rstrip()
    s=s.lstrip()
    s2=""
    for i,c in enumerate(s):
        if c==" " and ('0'<=s[i-1]<='9' or s[i-1]=='.'):
            if s[i+1].isalpha() or '0'<=s[i+1]<='9' or s[i+1]=='.':
                continue
        s2+=c
    return s2

def latex_asciimath(l):
    # if string is empty, do not bother
    if not l:
        return ''
    if len(l) == 1:
        return l
    payload['latex']=l
    try:
        r = requests.post(url, data=json.dumps(payload), headers=headers)
        if(r.status_code == requests.codes.ok):
            return json.loads(r.text)['asciimath'].strip()
    except requests.exceptions.RequestException as e:  # all exception
        print e
    return ''

