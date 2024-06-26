model_name="idefics1"
# model_name="llava"
# model_name="llava_next"
# model_name="idefics2"
# model_name="cogvlm"
# model_name="fuyu"
# model_name="kosmos2"
# model_name="openflamingo"
# model_name="otterimage"


bench_name="video_feedback"
# bench_name="eval_crafter"
# bench_name="genaibench"
# bench_name="vbench"


mkdir -p "./eval_results/${bench_name}"

data_repo_name="TIGER-Lab/VideoScore-Bench"
name_postfixs="[${bench_name}]"
result_file="./eval_results/${bench_name}/eval_${bench_name}_${model_name}.json"

### please select the avaiable GPU on your device
CUDA_VISIBLE_DEVICES=0 python eval_other_mllm.py \
    --data_repo_name $data_repo_name \
    --name_postfixs $name_postfixs \
    --result_file $result_file \
    --model_name $model_name \
    --bench_name $bench_name