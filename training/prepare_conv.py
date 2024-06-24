import json
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset

anno_data=load_dataset("TIGER-Lab/VideoFeedback",name="annotated",split="train")
real_data=load_dataset("TIGER-Lab/VideoFeedback",name="real",split="train")

all_data = []
for parquet in [anno_data,real_data]:
    for idx,item in tqdm(enumerate(parquet)):
        # "p110367_22.jpg"
        item['images'] = ["images/" + item['images'][0].split("_")[0] + "/" + image for i, image in enumerate(item['images'])]
        assert all([Path(image).exists() for image in item['images']]), item['images']
        all_data.append({
            "id": item['id'],
            "images": item['images'],
            "conversations": item['conversations'],
        })
        
with open("train_conv.json", "w") as f:
    json.dump(all_data, f, indent=4)