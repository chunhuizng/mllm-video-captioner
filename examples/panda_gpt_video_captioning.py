import torch
from lavis.models import load_model


model = load_model(name="panda_gpt", model_type="vicuna_7b")

text_list=["A dog.", "A car", "A bird", "A man"]
video_paths = ["/home/anonymous/new_ssd/webvid2m/val_dataset_downsampled/00000/00000704.mp4",
               "/home/anonymous/new_ssd/webvid2m/val_dataset_downsampled/00000/00000705.mp4",
               "/home/anonymous/new_ssd/webvid2m/val_dataset_downsampled/00000/00000706.mp4",
               "/home/anonymous/new_ssd/webvid2m/val_dataset_downsampled/00000/00000707.mp4",]
samples = {
    "image": video_paths,
    "text_input": text_list
}

out = model.generate(samples)
print(out)