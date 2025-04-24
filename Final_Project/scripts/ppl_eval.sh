cd /home/fanz2/SPARC/evaluation/pg19

MODELPATH=lmsys/longchat-7b-v1.5-32k
OUTPUT_DIR=results/ppl_eval/longchat
mkdir -p $OUTPUT_DIR

budget=1000000
start_idx=0
min_tokens=4096

CUDA_VISIBLE_DEVICES=0,1 python -u ppl_eval.py \
    --model_name_or_path $MODELPATH \
    --output_dir $OUTPUT_DIR \
    --start_idx $start_idx \
    --min_tokens $min_tokens \
    --num_eval_tokens 32000 \
    --quest --token_budget $budget --chunk_size 16 
