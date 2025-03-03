models=("DeepSeek-R1-Distill-Qwen-7B")

wbits=4
method=gptq
group_size=128
eval_framework=lmeval
eval_tasks=ARC_CHALLENGE
Eval=/data/disk0/Eval

for model in "${models[@]}"; do
    echo "Processing model: $model"
    
    Model_path="/data/disk0/Workspace/wdk/GPTQModel/$model"
    
    CUDA_LAUNCH_BLOCKING=1 python GPTQModel.py \
        --model "$Model_path" \
        --wbits $wbits --sym --group_size $group_size --compile \
        --device cuda:3 \
        --eval --eval_framework $eval_framework --eval_tasks $eval_tasks \
        --output_file "$Eval/${model}_${eval_framework}_result"  > "log/${model}.log" 2>&1
    
    echo "Finished processing $model"
done
