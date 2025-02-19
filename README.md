# Pretrained Image-Text Models are Secretly Video Captioners

This study adapts a pretrained image-text model, BLIP-2, to the task of video captioning. By using the visual and language understanding capabilities of BLIP-2, which is initially trained for image-text tasks, we show that the model also can effectively generate video captions without requiring architectural modifications or additional parameters. This approach enables high-quality caption generation by repurposing an existing image-text model for multi-frame video data.


## Installation

```bash
conda create -n video_cap python=3.8
conda activate video_cap
pip install -e .

pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
```

## Data Preparation
Please follow instructions from [here](LAVIS/lavis/datasets/download_scripts at main · anonymous/LAVIS) to download MSRVTT and MSVD datasets. VATEX datasets can be downloaded from its [official website](https://eric-xw.github.io/vatex-website/download.html).

## Cross-Entropy Training
We provide example script for training (MSRVTT and MSVD):
```bash
python -m torch.distributed.run --nproc_per_node=8 train.py --cfg-path lavis/projects/blip2/train/caption_msrvtt_flant5xl_ft.yaml
python -m torch.distributed.run --nproc_per_node=8 train.py --cfg-path lavis/projects/blip2/train/caption_msvd_flant5xl_ft.yaml
```

## SCST Training
We provide example script for training (MSRVTT and MSVD):
```bash
python -m torch.distributed.run --nproc_per_node=8 train.py --cfg-path lavis/projects/blip2/train/caption_msrvtt_flant5xl_scst.yaml
python -m torch.distributed.run --nproc_per_node=8 train.py --cfg-path lavis/projects/blip2/train/caption_msvd_flant5xl_scst.yaml
```

Full training configurations can be found in `lavis/projects/blip2/train`.

```
caption_msrvtt_flant5xl_ft.yaml
caption_msrvtt_flant5xl_scst.yaml
caption_msrvtt_opt2.7b_ft.yaml
caption_msrvtt_opt2.7b_scst.yaml
caption_msrvtt_opt6.7b_ft.yaml
caption_msrvtt_vicuna7b_ft.yaml
caption_msvd_flant5xl_ft.yaml
caption_msvd_flant5xl_scst.yaml
caption_vatex_flant5xl_ft.yaml
```

## Evaluation
We provide example scripts for evaluation (MSRVTT and MSVD):
```bash
python -m torch.distributed.run --nproc_per_node=8 evaluate.py --cfg-path lavis/projects/blip2/eval/caption_msrvtt_flant5xl_eval.yaml
python -m torch.distributed.run --nproc_per_node=8 evaluate.py --cfg-path lavis/projects/blip2/eval/caption_msvd_flant5xl_eval.yaml
```

## Pretrained Models
Pretrained models (weights) will be released soon.
