mkdir -p ./benchmark/eval_results/eval_crafter

## we use this variant "TIGER-Lab/MantisScore-anno-only" except for test set video_feedback
model_repo_name="TIGER-Lab/MantisScore-anno-only"

data_repo_name="TIGER-Lab/VideoFeedback-Bench"
frames_dir="./data/eval_crafter/test"
name_postfixs="['eval_crafter']"
result_file='./benchmark/eval_results/eval_crafter/eval_ec_mantisscore.json'
python benchmark/eval_mantis_score.py --model_repo_name $model_repo_name --data_repo_name $data_repo_name --frames_dir $frames_dir  --name_postfixs $name_postfixs --result_file $result_file 
