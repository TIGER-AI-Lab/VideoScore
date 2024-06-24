from huggingface_hub import hf_hub_download

hf_hub_download(repo_id="TIGER-Lab/VideoFeedback", filename="annotated/frames_annotated_train.zip", repo_type="dataset", local_dir="./data")
hf_hub_download(repo_id="TIGER-Lab/VideoFeedback", filename="annotated/frames_real_train.zip", repo_type="dataset", local_dir="./data")
