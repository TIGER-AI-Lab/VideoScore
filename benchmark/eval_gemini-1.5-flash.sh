PACKAGE_NAME=google.generativeai

if python -c "import $PACKAGE_NAME" &> /dev/null; then
    echo "$PACKAGE_NAME is already installed."
else
    echo "$PACKAGE_NAME is not installed. Installing..."
    pip install $PACKAGE_NAME
fi

base_model="gemini-1.5-flash-latest"

# # ===== eval MantisScore on VideoFeedback-test =====
# mkdir -p ./eval_results/video_feedback

# data_repo_name="TIGER-Lab/VideoFeedback-Bench"
# frames_dir="../data/video_feedback/test"
# name_postfixs="['video_feedback']"
# result_file="./eval_results/vbench/eval_vbench_${base_model}.json"
# bench_name="video_feedback"

# python eval_gemini.py \
#     --data_repo_name $data_repo_name \
#     --frames_dir $frames_dir \
#     --name_postfixs $name_postfixs \
#     --result_file $result_file \
#     --bench_name $bench_name




# # ===== eval MantisScore on EvalCrafter =====
# mkdir -p ./eval_results/eval_crafter

# data_repo_name="TIGER-Lab/VideoFeedback-Bench"
# frames_dir="../data/eval_crafter/test"
# name_postfixs="['eval_crafter']"
# result_file="./eval_results/vbench/eval_vbench_${base_model}.json"
# bench_name="eval_crafter"

# python eval_gemini.py \
#     --data_repo_name $data_repo_name \
#     --frames_dir $frames_dir \
#     --name_postfixs $name_postfixs \
#     --result_file $result_file \
#     --bench_name $bench_name




# # ===== eval MantisScore on GenAI-Bench =====
# mkdir -p ./eval_results/genaibench

# data_repo_name="TIGER-Lab/VideoFeedback-Bench"
# frames_dir="../data/genaibench/test"
# name_postfixs="['genaibench']"
# result_file="./eval_results/vbench/eval_vbench_${base_model}.json"
# bench_name="genaibench"

# python eval_gemini.py \
#     --data_repo_name $data_repo_name \
#     --frames_dir $frames_dir \
#     --name_postfixs $name_postfixs \
#     --result_file $result_file \
#     --bench_name $bench_name




# ===== eval MantisScore on VBench =====
mkdir -p ./benchmark/eval_results/vbench

data_repo_name="TIGER-Lab/VideoFeedback-Bench"
frames_dir="../data/vbench/test"
name_postfixs="['vbench']"
result_file="./eval_results/vbench/eval_vbench_${base_model}.json"
bench_name="vbench"

python eval_gemini.py \
    --data_repo_name $data_repo_name \
    --frames_dir $frames_dir \
    --name_postfixs $name_postfixs \
    --result_file $result_file \
    --bench_name $bench_name