source /home3/asrkws/shicheng2/bashrc_multimodal_kws
CUDA_VISIBLE_DEVICES=0,1 python train.py \
--lr 0.01 \
--use_bmuf \
--epochs 20 \
--batch_size 4 \
--train_snrs 3,6,9 \
--test_snrs 5,0,-5,-10 \
--optimizer SGD \
--network 'TVA_KWS_PLCL_AVmask' \
--datalist_dir '/work2/asrkws/shicheng2/Multimodal_KWS/data_list' \
--train_csv 'train_debug' \
--eval_csv 'eval_debug_inset,eval_debug_outset' \
--prob_addNoise 0.6 \
--lr_half_epochs 8,12,16 \
--out_dir './train_debug/model/' \
--log_path './train_debug/0_train.log' \
--display 1 \
--maxlen_text 40 \
--maxlen_vide 50 \
--maxlen_audi 100 \
# --train_csv 'train' \
# --train_csv 'checkout' \
