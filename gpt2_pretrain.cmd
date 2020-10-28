python pretrain_transformers.py ^
    --output_dir=D:/Proj_One/GitSeparate/ru-gpts/rugpt2large ^
    --model_type=gpt2 ^
    --model_name_or_path=./gpt2_large_bbpe_v50 ^
    --do_train ^
    --train_data_file=all_essays.txt ^
    --do_eval ^
    --eval_data_file=valid_essays.txt ^
    --fp16 ^
    --per_gpu_train_batch_size 1 ^
    --per_gpu_eval_batch_size 1 ^
    --gradient_accumulation_steps 1 ^
    --line_by_line