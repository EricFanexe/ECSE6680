cd /home/fanz2/SPARC/evaluation/LongBench/

model="longchat-v1.5-7b-32k"

# for task in "hotpotqa" "lcc" "narrativeqa" "passage_retrieval_en" "qasper" "samsum" "triviaqa" "gov_report"
for task in "lcc" "narrativeqa"
do
    # CUDA_VISIBLE_DEVICES=0,1 python -u pred.py \
    #     --model $model --task $task

    for budget in 4096
    do
        CUDA_VISIBLE_DEVICES=0,1 python -u pred.py \
            --model $model --task $task \
            --quest --token_budget $budget --chunk_size 32
    done
done

# CUDA_VISIBLE_DEVICES=0,1 python -u eval.py --model $model
