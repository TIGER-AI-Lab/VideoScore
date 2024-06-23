data_repo_name="TIGER-Lab/VideoFeedback-Bench"
result_dir='./benchmark/eval_results/genaibench'
acc_output_file="./benchmark/eval_results/genaibench/genaibench_pairwise_acc.txt"
python benchmark/get_genaibench_pairwise_acc.py --data_repo_name $data_repo_name --result_dir $result_dir > $acc_output_file