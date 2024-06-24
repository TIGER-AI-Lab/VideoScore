
bench_name="video_feedback"
# bench_name="eval_crafter"
# bench_name="genaibench"
# bench_name="vbench"

mkdir -p mkdir -p "./eval_results/${bench_name}"

## for video_feedback, we use default model of VideoScore; 
## while for the other three test sets, we use the variant VideoScore-anno-only, with real videos excluded from training set
if [ "$metric_name" = "video_feedback" ]; then
    model_repo_name="TIGER-Lab/VideoScore"
else
    model_repo_name="TIGER-Lab/VideoScore-anno-only"
fi


data_repo_name="TIGER-Lab/VideoScore-Bench"
frames_dir="../data/${bench_name}/test"
name_postfixs="[${bench_name}]"
result_file="./eval_results/${bench_name}/eval_${bench_name}_videoscore.json"

python eval_videoscore.py --model_repo_name $model_repo_name \
    --data_repo_name $data_repo_name \
    --frames_dir $frames_dir  \
    --name_postfixs $name_postfixs \
    --result_file $result_file \
    --bench_name $bench_name \


