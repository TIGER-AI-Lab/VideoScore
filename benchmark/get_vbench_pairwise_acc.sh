data_repo_name="TIGER-Lab/VideoFeedback-Bench"
result_dir='./eval_results/vbench'
acc_output_file="./eval_results/vbench/vbench_pairwise_acc.txt"
python get_vbench_pairwise_acc.py --data_repo_name $data_repo_name --result_dir $result_dir > $acc_output_file

# python get_vbench_pairwise_acc.py --data_repo_name $data_repo_name --result_dir $result_dir