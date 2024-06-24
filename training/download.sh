
mkdir -p ./data

if [ -d "${bench_name}/${split}/frames_${postfix}" ]; then
    echo "frames exists"
else
    echo "frames not exist, downloading..."
    wget "https://huggingface.co/datasets/TIGER-Lab/VideoFeedback/resolve/main/real/frames_real_train.zip" -O "./data/frames_real_train.zip"
    unzip "./data/frames_real_train.zip" -d "./data/images/"
    rm "./data/frames_real_train.zip"    
fi