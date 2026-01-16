## Single LLM call

Test client 
```
python ipmai/sing_llm_call/client.py
```




# 前台运行
python ipmai/sing_llm_call/run_single_llm_call.py \
  --output_dir "/Users/kevin/project_python/Pap2Pat/outputs/single-llm-call/qwen3" \
  --model_profile qwen3 \
  --split test \
  --outline_suffix long \
  --temperature 0.3


# 后台运行（
nohup python ipmai/sing_llm_call/run_single_llm_call.py \
  --output_dir "/Users/kevin/project_python/Pap2Pat/outputs/single-llm-call/qwen3" \
  --model_profile qwen3 \
  --split test \
  --outline_suffix long \
  --temperature 0.3 \
  > ./qwen3.log3 2>&1 &

nohup python ipmai/sing_llm_call/run_single_llm_call.py \
  --output_dir "/Users/kevin/project_python/Pap2Pat/outputs/single-llm-call/qwen3max" \
  --model_profile qwen3-max \
  --split test \
  --outline_suffix long \
  --temperature 0.3 \
  > ./qwen3max.log 2>&1 &

nohup python ipmai/sing_llm_call/run_single_llm_call.py \
  --output_dir "/Users/kevin/project_python/Pap2Pat/outputs/single-llm-call/deepseek-v3" \
  --model_profile deepseek-v3 \
  --split test \
  --outline_suffix long \
  --temperature 0.3 \
  > ./deepseek-v3.log 2>&1 &

```


Run with empty outline
```
nohup python ipmai/sing_llm_call/run_single_llm_call.py \
  --output_dir "/Users/kevin/project_python/Pap2Pat/outputs/single-llm-call/qwen3-emptyoutline" \
  --model_profile qwen3 \
  --split test \
  --outline_suffix empty \
  --temperature 0.3 \
  > ./qwen3.log4 2>&1 &


nohup python ipmai/sing_llm_call/run_single_llm_call.py \
  --output_dir "/Users/kevin/project_python/Pap2Pat/outputs/single-llm-call/qwen3max-emptyoutline" \
  --model_profile qwen3-max \
  --split test \
  --outline_suffix empty \
  --temperature 0.3 \
  > ./qwen3max.log3 2>&1 &


nohup python ipmai/sing_llm_call/run_single_llm_call.py \
  --output_dir "/Users/kevin/project_python/Pap2Pat/outputs/single-llm-call/deepseek-v3-emptyoutline" \
  --model_profile deepseek-v3 \
  --split test \
  --outline_suffix empty \
  --temperature 0.3 \
  > ./deepseek-v3.log3 2>&1 &


nohup python ipmai/sing_llm_call/run_single_llm_call.py \
  --output_dir "/Users/kevin/project_python/Pap2Pat/outputs/single-llm-call/qwen2" \
  --model_profile qwen2 \
  --split test \
  --outline_suffix empty \
  --temperature 0.3 \
  --max_tokens 8000 \
  > ./qwen2.log5 2>&1 &

```






# Evaluate Result

```
#RESULT_PATH=qwen3/pred_test
#RESULT_PATH=qwen3max/pred_test
#RESULT_PATH=ollama-qwen2.5/predictions
RESULT_PATH=deepseek-v3/pred_test

# 后台运行评估（推荐）
nohup python ipmai/evaluate/simple_metrics.py \
  --pred_dir "/Users/kevin/project_python/Pap2Pat/outputs/single-llm-call/$RESULT_PATH" \
  > ./eval_simple_qwen3.log 2>&1 &

nohup python ipmai/evaluate/token_metrics.py \
  --pred_dir "/Users/kevin/project_python/Pap2Pat/outputs/single-llm-call/$RESULT_PATH" \
  --tokenizer hf --model meta-llama/Meta-Llama-3-8B-Instruct \
  > ./eval_token.log 2>&1 &

nohup python ipmai/evaluate/embedding_metrics.py \
  --pred_dir "/Users/kevin/project_python/Pap2Pat/outputs/single-llm-call/$RESULT_PATH" \
  --backend api \
  --base_url https://dashscope.aliyuncs.com/compatible-mode/v1 \
  --api_key "$BAILIAN_API_KEY" \
  --model text-embedding-v4 \
  > ./eval_embedding.log 2>&1 &

# 查看进度
tail -f ./eval_simple.log
tail -f ./eval_token.log
tail -f ./eval_embedding.log
```



## Summary the result 

cd outputs/single-llm-call
for csv in *-res.csv; do
  python ../../ipmai/evaluate/summarize_metrics.py \
    --csv "$csv" \
    --output "${csv%.csv}-summary.txt"
done





