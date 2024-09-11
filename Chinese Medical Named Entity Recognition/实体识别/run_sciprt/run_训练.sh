lr=3e-5
epoch=3

rdrop_type="softmax"
rdrop_alpha=2
pretrained_model_name=RoBERTa_zh_Large_PyTorch
data_dir=./CMeIE
gpu=0
cd ../

echo "Now learning rate:${lr}"
version1=$(command date +%m-%d-%H-%M --date="+8 hour")
echo "version1:${version1}"
python -u ./code_main_zh/run_gper.py --rdrop_alpha ${rdrop_alpha} --do_rdrop --rdrop_type ${rdrop_type} --max_length 256 --inner_dim 64 --epochs ${epoch} --do_train --learning_rate ${lr} --train_batch_size 24 --eval_batch_size 512 --time ${version1} --pretrained_model_name ${pretrained_model_name} --data_dir ${data_dir} --devices ${gpu}

