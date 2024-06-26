cd ./data


# bench_name=video_feedback
# bench_name=genaibench
bench_name=eval_crafter
# bench_name=vbench

repo_id=TIGER-Lab/VideoScore-Bench

mkdir -p $bench_name
mkdir -p "${bench_name}/test"


if [ -d "${bench_name}/${split}/frames_${postfix}" ]; then
    echo "frames dir exists"
else
    echo "frames dir not exist, downloading..."
    wget wget "https://huggingface.co/datasets/${repo_id}/resolve/main/${bench_name}/frames_${bench_name}_test.zip" -O "./${bench_name}/test/frames_${bench_name}_test.zip"
    unzip "./${bench_name}/test/frames_${bench_name}_test.zip" -d "./${bench_name}/test/frames_${bench_name}"
    rm "./${bench_name}/test/frames_${bench_name}_test.zip"     
fi
