# preprocess
#~/miniconda2/bin/python preprocess.py -data_type img -src_dir data/im2text/hw/images_processed/ -train_src data/im2text/hw/src-train.txt -train_tgt data/im2text/hw/tgt-train.txt -valid_src data/im2text/hw/src-val.txt -valid_tgt data/im2text/hw/tgt-val.txt -save_data data/im2text/hw

# train
~/miniconda2/bin/python train.py -model_type img -data data/im2text/hw -save_model im2text-hw-model -gpuid 0 -batch_size 20 -max_grad_norm 20 -optim adam -learning_rate 0.001 -epochs 30

# debug
#~/miniconda2/bin/python -m pdb train.py -model_type img -data data/im2text/hw -save_model im2text-hw-model -gpuid 0 -batch_size 20 -max_grad_norm 20
# save
#~/miniconda2/bin/python train.py -model_type img -data data/im2text/hw -save_model im2text-hw-model -gpuid 0 -batch_size 32 -max_grad_norm 20 -optim adam -learning_rate 0.001 -batch_size 20


