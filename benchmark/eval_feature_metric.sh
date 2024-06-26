
# metric_name="PIQE"
# metric_name="BRISQUE"
# metric_name="CLIP-sim"
# metric_name="DINO-sim"
# metric_name="SSIM-sim"
# metric_name="MSE-dyn"
# metric_name="SSIM-dyn"
# metric_name="CLIP-Score"
metric_name="X-CLIP-Score"

# bench_name="video_feedback"
# bench_name="eval_crafter"
bench_name="genaibench"
# bench_name="vbench"



if [ "$metric_name" = "PIQE" -o "$metric_name" = "BRISQUE" ]; then
    if conda info --envs | grep -q "piqe"; then
        echo "env piqe exists"
    else
        echo "env piqe not exist, creating..."
        conda create -n piqe
    fi
    conda activate piqe
    pip install pypiqe
fi


mkdir -p "./eval_results/${bench_name}"

data_repo_name="TIGER-Lab/VideoScore-Bench"
frames_dir="../data/${bench_name}/test"
name_postfixs="[${bench_name}]"
result_file="./eval_results/${bench_name}/eval_${bench_name}_${metric_name}.json"


python eval_feature_metric.py \
    --data_repo_name $data_repo_name \
    --metric_name $metric_name \
    --bench_name $bench_name \
    --frames_dir $frames_dir  \
    --name_postfixs $name_postfixs \
    --result_file $result_file


if [ "$metric_name" = "PIQE" -o "$metric_name" = "BRISQUE" ]; then
    conda deactivate
fi
