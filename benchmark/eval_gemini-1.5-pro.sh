PACKAGE_NAME=google.generativeai

if python -c "import $PACKAGE_NAME" &> /dev/null; then
    echo "$PACKAGE_NAME is already installed."
else
    echo "$PACKAGE_NAME is not installed. Installing..."
    pip install $PACKAGE_NAME
fi

base_model="gemini-1.5-pro-latest"

# # ===== eval MantisScore on VideoFeedback-test =====
# mkdir -p ./eval_results/video_feedback

# data_repo_name="TIGER-Lab/VideoFeedback-Bench"
# bench_name="video_feedback"
# frames_dir="../data/${bench_name}/test"
# name_postfixs="[${bench_name}]"
# result_file="./eval_results/${bench_name}/eval_${bench_name}_mantisscore.json"


# python eval_gemini.py \
#     --data_repo_name $data_repo_name \
#     --frames_dir $frames_dir \
#     --name_postfixs $name_postfixs \
#     --result_file $result_file \
#     --bench_name $bench_name




# # ===== eval MantisScore on EvalCrafter =====
# mkdir -p ./eval_results/eval_crafter

# data_repo_name="TIGER-Lab/VideoFeedback-Bench"
# bench_name="eval_crafter"
# frames_dir="../data/${bench_name}/test"
# name_postfixs="[${bench_name}]"
# result_file="./eval_results/${bench_name}/eval_${bench_name}_mantisscore.json"

# python eval_gemini.py \
#     --data_repo_name $data_repo_name \
#     --frames_dir $frames_dir \
#     --name_postfixs $name_postfixs \
#     --result_file $result_file \
#     --bench_name $bench_name




# # ===== eval MantisScore on GenAI-Bench =====
# mkdir -p ./eval_results/genaibench

# data_repo_name="TIGER-Lab/VideoFeedback-Bench"
# bench_name="genaibench"
# frames_dir="../data/${bench_name}/test"
# name_postfixs="[${bench_name}]"
# result_file="./eval_results/${bench_name}/eval_${bench_name}_mantisscore.json"

# python eval_gemini.py \
#     --data_repo_name $data_repo_name \
#     --frames_dir $frames_dir \
#     --name_postfixs $name_postfixs \
#     --result_file $result_file \
#     --bench_name $bench_name




# ===== eval MantisScore on VBench =====
mkdir -p ./benchmark/eval_results/vbench

data_repo_name="TIGER-Lab/VideoFeedback-Bench"
bench_name="vbench"
frames_dir="../data/${bench_name}/test"
name_postfixs="[${bench_name}]"
result_file="./eval_results/${bench_name}/eval_${bench_name}_mantisscore.json"

python eval_gemini.py \
    --data_repo_name $data_repo_name \
    --frames_dir $frames_dir \
    --name_postfixs $name_postfixs \
    --result_file $result_file \
    --bench_name $bench_name