python pretrain_transformers.py ^
    --output_dir=D:/Proj_One/GitSeparate/ru-gpts/words_model2 ^
    --model_type=gpt2 ^
    --model_name_or_path=D:/Proj_One/GitSeparate/ru-gpts/words_model ^
    --do_train ^
    --train_data_file=words_train.txt ^
    --do_eval ^
    --eval_data_file=words_valid.txt ^
    --per_gpu_train_batch_size 1 ^
    --gradient_accumulation_steps 1 ^
    --per_gpu_eval_batch_size 1 ^
    --num_train_epochs 5 ^
    --block_size 2048 ^
    --overwrite_output_dir ^
    --fp16 ^
    --fp16_opt_level=O1
    