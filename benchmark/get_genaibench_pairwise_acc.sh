data_repo_name="TIGER-Lab/VideoScore-Bench"
result_dir='./eval_results/genaibench'
acc_output_file="./eval_results/genaibench/genaibench_pairwise_acc.txt"
python get_genaibench_pairwise_acc.py --data_repo_name $data_repo_name --result_dir $result_dir > $acc_output_file