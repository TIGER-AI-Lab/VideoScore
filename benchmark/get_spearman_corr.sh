bench_name="video_feedback"
# bench_name="eval_crafter"

result_dir="./eval_results/${bench_name}"
python get_spearman_corr.py --result_dir $result_dir --bench_name $bench_name