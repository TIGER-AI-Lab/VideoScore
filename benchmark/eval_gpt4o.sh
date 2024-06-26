bench_name="video_feedback"
# bench_name="eval_crafter"
# bench_name="genaibench"
# bench_name="vbench"


mkdir -p "./eval_results/${bench_name}"

data_repo_name="TIGER-Lab/VideoScore-Bench"
name_postfixs="[${bench_name}]"
result_file="././eval_results/${bench_name}/eval_${bench_name}_gpt4o.json"

python eval_gpt4o.py \
    --data_repo_name $data_repo_name \
    --name_postfixs $name_postfixs \
    --result_file $result_file \
    --bench_name $bench_name