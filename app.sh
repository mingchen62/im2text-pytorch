#!/bin/bash
python app.py -model im2text-hw-model_acc_96.75_ppl_1.16_e13.pt  -data_type img -replace_unk -verbose -n_best 3 -batch_size 1 

