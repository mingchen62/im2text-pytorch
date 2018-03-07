#~/miniconda2/bin/python predict.py -model im2text-hw-model_acc_96.75_ppl_1.16_e13.pt  -data_type img   -src_dir  data/im2text/hw/scg -src data/im2text/hw/src-test2.txt  -output predem.txt -replace_unk -verbose -n_best 3 -batch_size 1 -gpu 0 
~/miniconda2/bin/python predict.py -model im2text-hw-model_acc_96.75_ppl_1.16_e13.pt  -data_type img -replace_unk -verbose -n_best 3 -batch_size 1 -gpu 0 
