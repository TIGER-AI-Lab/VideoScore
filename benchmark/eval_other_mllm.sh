model_name="idefics1"
# model_name="llava"
# model_name="llava_next"
# model_name="idefics2"
# model_name="cogvlm"
# model_name="fuyu"
# model_name="kosmos2"
# model_name="openflamingo"
# model_name="otterimage"


# ===== eval MantisScore on VideoFeedback-test =====
mkdir -p ./eval_results/video_feedback

data_repo_name="TIGER-Lab/VideoScore-Bench"
bench_name="video_feedback"
name_postfixs="[${bench_name}]"
result_file="./eval_results/${bench_name}/eval_${bench_name}_mantisscore.json"

python eval_other_mllm.py \
    --data_repo_name $data_repo_name \
    --name_postfixs $name_postfixs \
    --result_file $result_file \
    --model_name $model_name \
    --bench_name $bench_name





# # ===== eval MantisScore on EvalCrafter =====
# mkdir -p ./eval_results/eval_crafter

# data_repo_name="TIGER-Lab/VideoScore-Bench"
# bench_name="eval_crafter"
# name_postfixs="[${bench_name}]"
# result_file="./eval_results/${bench_name}/eval_${bench_name}_mantisscore.json"

# python eval_other_mllm.py \
#     --data_repo_name $data_repo_name \
#     --name_postfixs $name_postfixs \
#     --result_file $result_file \
#     --model_name $model_name \
#     --bench_name $bench_name




# # ===== eval MantisScore on GenAI-Bench =====
# mkdir -p ./eval_results/genaibench

# data_repo_name="TIGER-Lab/VideoScore-Bench"
# bench_name="genaibench"
# name_postfixs="[${bench_name}]"
# result_file="./eval_results/${bench_name}/eval_${bench_name}_mantisscore.json"

# python eval_other_mllm.py \
#     --data_repo_name $data_repo_name \
#     --name_postfixs $name_postfixs \
#     --result_file $result_file \
#     --model_name $model_name \
#     --bench_name $bench_name




# # ===== eval MantisScore on VBench =====
# mkdir -p ./benchmark/eval_results/vbench

# data_repo_name="TIGER-Lab/VideoScore-Bench"
# bench_name="vbench"
# name_postfixs="[${bench_name}]"
# result_file="./eval_results/${bench_name}/eval_${bench_name}_mantisscore.json"

# python eval_other_mllm.py \
#     --data_repo_name $data_repo_name \
#     --name_postfixs $name_postfixs \
#     --result_file $result_file \
#     --model_name $model_name \
#     --bench_name $bench_name