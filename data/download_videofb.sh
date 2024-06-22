cd ./data

curr_dir=videofb
repo_id=TIGER-Lab/VideoFeedback

mkdir -p $curr_dir
mkdir -p "${curr_dir}/train"
mkdir -p "${curr_dir}/test"

splits=("train" "test")
file_postfixs=("annotated" "real")

for split in "${splits[@]}"; do
    for postfix in "${file_postfixs[@]}"; do
        if [ -e "${curr_dir}/${split}/data_${postfix}.json" ]; then
            echo "json file exists"
        else
            echo "file not exist, downloading..."
            wget wget "https://huggingface.co/datasets/${repo_id}/resolve/main/${split}/data_${postfix}.json" -O "./${curr_dir}/${split}/data_${postfix}.json"
        fi

        if [ -d "${curr_dir}/${split}/frames_${postfix}" ]; then
            echo "frames dir exists"
        else
            echo "frames dir not exist, downloading..."
            wget wget "https://huggingface.co/datasets/${repo_id}/resolve/main/${split}/frames_${postfix}.zip" -O "./${curr_dir}/${split}/frames_${postfix}.zip"
            unzip "./${curr_dir}/${split}/frames_${postfix}.zip" -d "./${curr_dir}/${split}/frames_${postfix}"
            rm "./${curr_dir}/${split}/frames_${postfix}.zip"      
        fi
    done
done

