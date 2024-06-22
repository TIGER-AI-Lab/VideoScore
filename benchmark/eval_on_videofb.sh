model_name="TIGER-Lab/MantisScore"
data_dir="./data/videofb/test"
name_postfixs="['annotated','real']"
result_file='./benchmark/eval_results/eval_videofb_mantisscore.json'

python benchmark/eval_on_videofb.py --model_name $model_name --data_dir $data_dir  --name_postfixs $name_postfixs --result_file $result_file 