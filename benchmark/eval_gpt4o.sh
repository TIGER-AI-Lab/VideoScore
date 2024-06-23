# # ===== eval MantisScore on VideoFeedback-test =====
# mkdir -p ./eval_results/video_feedback

# data_repo_name="TIGER-Lab/VideoFeedback-Bench"
# name_postfixs="['video_feedback']"
# result_file='./eval_results/video_feedback/eval_video_feedback_mantisscore.json'
# bench_name="video_feedback"

# python eval_gpt4o.py \
#     --data_repo_name $data_repo_name \
#     --name_postfixs $name_postfixs \
#     --result_file $result_file \
#     --bench_name $bench_name




# # ===== eval MantisScore on EvalCrafter =====
# mkdir -p ./eval_results/eval_crafter

# data_repo_name="TIGER-Lab/VideoFeedback-Bench"
# name_postfixs="['eval_crafter']"
# result_file='./eval_results/eval_crafter/eval_ec_mantisscore.json'
# bench_name="eval_crafter"

# python eval_gpt4o.py \
#     --data_repo_name $data_repo_name \
#     --name_postfixs $name_postfixs \
#     --result_file $result_file \
#     --bench_name $bench_name




# ===== eval MantisScore on GenAI-Bench =====
mkdir -p ./eval_results/genaibench

data_repo_name="TIGER-Lab/VideoFeedback-Bench"
name_postfixs="['genaibench']"
result_file='./eval_results/genaibench/eval_genaibench_mantisscore.json'
bench_name="genaibench"

python eval_gpt4o.py \
    --data_repo_name $data_repo_name \
    --name_postfixs $name_postfixs \
    --result_file $result_file \
    --bench_name $bench_name




# # ===== eval MantisScore on VBench =====
# mkdir -p ./benchmark/eval_results/vbench

# data_repo_name="TIGER-Lab/VideoFeedback-Bench"
# name_postfixs="['vbench']"
# result_file='./eval_results/vbench/eval_vbench_mantisscore.json'
# bench_name="vbench"

# python eval_gpt4o.py \
#     --data_repo_name $data_repo_name \
#     --name_postfixs $name_postfixs \
#     --result_file $result_file \
#     --bench_name $bench_name