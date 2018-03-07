~/miniconda2/bin/python translate.py -model im2text-hw-model_acc_96.75_ppl_1.16_e13.pt  -data_type img   -src_dir  data/im2text/hw/images_processed -src data/im2text/hw/src-test.txt -tgt data/im2text/hw/tgt-test.txt  -output prede13.txt -replace_unk -verbose -n_best 3 -batch_size 20
#~/miniconda2/bin/python translate.py -model im2text-hw-model_acc_96.75_ppl_1.16_e12.pt  -data_type img   -src_dir  data/im2text/hw/images_processed -src data/im2text/hw/src-test.txt -tgt data/im2text/hw/tgt-test.txt  -output pred.txt -replace_unk -verbose -n_best 3 -batch_size 20

