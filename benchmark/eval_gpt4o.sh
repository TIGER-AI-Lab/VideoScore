# # ===== eval MantisScore on VideoFeedback-test =====
# mkdir -p ./eval_results/video_feedback

# data_repo_name="TIGER-Lab/VideoFeedback-Bench"
# bench_name="video_feedback"
# name_postfixs="[${bench_name}]"
# result_file="./eval_results/${bench_name}/eval_${bench_name}_mantisscore.json"

# python eval_gpt4o.py \
#     --data_repo_name $data_repo_name \
#     --name_postfixs $name_postfixs \
#     --result_file $result_file \
#     --bench_name $bench_name




# # ===== eval MantisScore on EvalCrafter =====
# mkdir -p ./eval_results/eval_crafter

# data_repo_name="TIGER-Lab/VideoFeedback-Bench"
# bench_name="eval_crafter"
# name_postfixs="[${bench_name}]"
# result_file="./eval_results/${bench_name}/eval_${bench_name}_mantisscore.json"


# python eval_gpt4o.py \
#     --data_repo_name $data_repo_name \
#     --name_postfixs $name_postfixs \
#     --result_file $result_file \
#     --bench_name $bench_name




# ===== eval MantisScore on GenAI-Bench =====
mkdir -p ./eval_results/genaibench

data_repo_name="TIGER-Lab/VideoFeedback-Bench"
bench_name="genaibench"
name_postfixs="[${bench_name}]"
result_file="./eval_results/${bench_name}/eval_${bench_name}_mantisscore.json"

python eval_gpt4o.py \
    --data_repo_name $data_repo_name \
    --name_postfixs $name_postfixs \
    --result_file $result_file \
    --bench_name $bench_name




# # ===== eval MantisScore on VBench =====
# mkdir -p ./benchmark/eval_results/vbench

# data_repo_name="TIGER-Lab/VideoFeedback-Bench"
# bench_name="vbench"
# name_postfixs="[${bench_name}]"
# result_file="./eval_results/${bench_name}/eval_${bench_name}_mantisscore.json"

# python eval_gpt4o.py \
#     --data_repo_name $data_repo_name \
#     --name_postfixs $name_postfixs \
#     --result_file $result_file \
#     --bench_name $bench_name