
wget "https://huggingface.co/datasets/TIGER-Lab/VideoFeedback/resolve/main/real/frames_real_train.zip" -O "./frames_real_train.zip"
unzip "./frames_real_train.zip" -d "./images/"
rm "./frames_real_train.zip"    

wget "https://huggingface.co/datasets/TIGER-Lab/VideoFeedback/resolve/main/annotated/frames_annotated_train.zip" -O "./frames_annotated_train.zip"
unzip "./frames_annotated_train.zip" -d "./images/"
rm "./frames_annotated_train.zip"    