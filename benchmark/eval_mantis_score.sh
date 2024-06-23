# # ===== eval MantisScore on VideoFeedback-test =====
# mkdir -p ./eval_results/video_feedback

# model_repo_name="TIGER-Lab/MantisScore"
# data_repo_name="TIGER-Lab/VideoFeedback-Bench"
# frames_dir="../data/video_feedback/test"
# name_postfixs="['video_feedback']"
# result_file='./eval_results/video_feedback/eval_video_feedback_mantisscore.json'
# bench_name="video_feedback"

# CUDA_VISIBLE_DEVICES=0 python eval_mantis_score.py --model_repo_name $model_repo_name \
#     --data_repo_name $data_repo_name \
#     --frames_dir $frames_dir  \
#     --name_postfixs $name_postfixs \
#     --result_file $result_file \
#     --bench_name $bench_name \




# # ===== eval MantisScore on EvalCrafter =====
# mkdir -p ./eval_results/eval_crafter

# # we use this variant "TIGER-Lab/MantisScore-anno-only" except for VideoFeedback-test
# model_repo_name="TIGER-Lab/MantisScore-anno-only"
# data_repo_name="TIGER-Lab/VideoFeedback-Bench"
# frames_dir="../data/eval_crafter/test"
# name_postfixs="['eval_crafter']"
# result_file='./eval_results/eval_crafter/eval_ec_mantisscore.json'
# bench_name="eval_crafter"

# CUDA_VISIBLE_DEVICES=1 python eval_mantis_score.py --model_repo_name $model_repo_name \
    # --data_repo_name $data_repo_name \
    # --frames_dir $frames_dir \
    # --name_postfixs $name_postfixs \
    # --result_file $result_file \
    # --bench_name $bench_name




# # ===== eval MantisScore on GenAI-Bench =====
# mkdir -p ./eval_results/genaibench

# # we use this variant "TIGER-Lab/MantisScore-anno-only" except for VideoFeedback-test
# model_repo_name="TIGER-Lab/MantisScore-anno-only"
# data_repo_name="TIGER-Lab/VideoFeedback-Bench"
# frames_dir="../data/genaibench/test"
# name_postfixs="['genaibench']"
# result_file='./eval_results/genaibench/eval_genaibench_mantisscore.json'
# bench_name="genaibench"

# CUDA_VISIBLE_DEVICES=2 python eval_mantis_score.py --model_repo_name $model_repo_name \
#     --data_repo_name $data_repo_name \
#     --frames_dir $frames_dir \
#     --name_postfixs $name_postfixs \
#     --result_file $result_file \
#     --bench_name $bench_name




# ===== eval MantisScore on VBench =====
mkdir -p ./benchmark/eval_results/vbench

# we use this variant "TIGER-Lab/MantisScore-anno-only" except for VideoFeedback-test
model_repo_name="TIGER-Lab/MantisScore-anno-only"
data_repo_name="TIGER-Lab/VideoFeedback-Bench"
frames_dir="../data/vbench/test"
name_postfixs="['vbench']"
result_file='./eval_results/vbench/eval_vbench_mantisscore.json'
bench_name="vbench"

python eval_mantis_score.py --model_repo_name $model_repo_name \
    --data_repo_name $data_repo_name \
    --frames_dir $frames_dir \
    --name_postfixs $name_postfixs \
    --result_file $result_file \
    --bench_name $bench_name