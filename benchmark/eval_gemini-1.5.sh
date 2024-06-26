PACKAGE_NAME=google.generativeai

if python -c "import $PACKAGE_NAME" &> /dev/null; then
    echo "$PACKAGE_NAME is already installed."
else
    echo "$PACKAGE_NAME is not installed. Installing..."
    pip install $PACKAGE_NAME
fi


# base_model="gemini-1.5-flash-latest"
base_model="gemini-1.5-pro-latest"


bench_name="video_feedback"
# bench_name="eval_crafter"
# bench_name="genaibench"
# bench_name="vbench"


mkdir -p "./eval_results/${bench_name}"

data_repo_name="TIGER-Lab/VideoScore-Bench"
frames_dir="../data/${bench_name}/test"
name_postfixs="[${bench_name}]"
result_file="./eval_results/${bench_name}/eval_${bench_name}_${base_model}.json"
python eval_gemini.py \
    --data_repo_name $data_repo_name \
    --bench_name $bench_name \
    --frames_dir $frames_dir \
    --name_postfixs $name_postfixs \
    --result_file $result_file


