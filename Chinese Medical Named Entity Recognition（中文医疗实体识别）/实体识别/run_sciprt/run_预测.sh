lr=3e-5
epoch=3

rdrop_type="softmax"
rdrop_alpha=2
pretrained_model_name=RoBERTa_zh_Large_PyTorch
data_dir=./CMeIE
gpu=0
cd ../

version1='08-25-22-34'

echo "开始预测"
python -u ./code_main_zh/run_gper.py --rdrop_alpha ${rdrop_alpha} --do_rdrop --rdrop_type ${rdrop_type} --epochs ${epoch} --learning_rate ${lr} --max_length 256 --inner_dim 64 --do_predict --eval_batch_size 512 --model_version ${version1} --pretrained_model_name ${pretrained_model_name} --data_dir ${data_dir} --devices ${gpu}

