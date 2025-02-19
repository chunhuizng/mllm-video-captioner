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

## More experiment details

We hope these results are useful to your experimental explorations. Feel free to discuss this with us (especially Yiren and me)!

| LLM Decoder | Description | Train ViT | Train LLM (LoRA) | Num of Frames | Video Resolution | SCST | CIDEr |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FLAN-T5-XL | No MSRVTT training, directly using BLIP-2 zero-shot results. | - | - | - | - | - | 50.8 (val) |
| FLAN-T5-XL | Best result with 8 frames, trained weights were later used for 8-frame RL training. | No | No | 8 | 224 | No | 73.6 |
| FLAN-T5-XL | LoRA LLM, trained with 8 frames (epoch-based → iteration-based), used WebVid, and adjusted hyperparameters. WebVid did not improve as expected or even at all, so we abandoned it in future experiments. | No | Yes | 8 | 224 | No | 73.0 |
| FLAN-T5-XL | LoRA, 8 frames (fine-tuned ViT, completely failed). | Yes | Yes | 8 | 224 | No | 61.8 |
| FLAN-T5-XL | LoRA, 8 frames (fine-tuned ViT for only 5 epochs, not too bad). | Yes | Yes | 8 | 224 | No | 71.4 |
| FLAN-T5-XL | LoRA, 8 frames (fine-tuned ViT, poor result, i.e., 68.4). | Yes | Yes | 8 | 224 | No | 68.4 |
| FLAN-T5-XL | No MSRVTT training, directly using BLIP-2 zero-shot results. | - | - | - | - | - | 50.8 (val) |
| FLAN-T5-XL | Train only Q-former. | No | No | 8 | 224 | No | 71.4 |
| FLAN-T5-XL | Train only Q-former, better hyper-parameters? | No | No | 8 | 224 | No | 73.2 |
| FLAN-T5-XL | Train only Q-former, conclusion: 4 frame results are worse than 8 frames. | No | No | 4 | 224 | No | 70.4 |
| FLAN-T5-XL | Train only Q-former, conclusion: 4 frame results are worse than 8 frames. | No | No | 4 | 224 | No | 71.7 |
| FLAN-T5-XL | Best result with 8 frames, trained weights used later for 8 frame RL training. | No | No | 8 | 224 | No | 73.6 |
| FLAN-T5-XL | SCST reinforcement learning: initialized with pre-trained model 20231210062. CIDEr: 77.4 | - | - | - | - | Yes | 77.4 |
| FLAN-T5-XL | SCST, changed a set of hyper-parameters: initialized with pre-trained model 20231210062. CIDEr: 77.4 | - | - | - | - | Yes | 77.4 |
| FLAN-T5-XL | Changed another set of hyper-parameters and trained 8 frame Q-former only once. | No | No | 8 | 224 | No | 71.8 |
| FLAN-T5-XL | First time training LoRA LLM, used 16 frames | No | Yes | 16 | 224 | No | 73.0 |
| FLAN-T5-XL | LoRA LLM, used 8 frames (seemingly not as good as 16 frames), used webvid (pseudo-labeled in msrvtt) mixed data | No | Yes | 8 | 224 | No | 71.9 |
| FLAN-T5-XL | LoRA LLM, used 8 frames (epoch based -> iter based), seems tuning hyper-parameters for LoRA can achieve 73+. Used pure webvid data | No | Yes | 8 | 224 | No | 73.7 |
| FLAN-T5-XL | LoRA LLM, used 8 frames (epoch based -> iter based), used webvid, changed hyper-params again. Feels like webvid did not improve as expected or even not at all, so it was dropped in future experiments. | No | Yes | 8 | 224 | No | 73.0 |
| FLAN-T5-XL | LoRA, 16 frames | No | Yes | 16 | 224 | No | 72.6 |
| FLAN-T5-XL | LoRA, 8 frames (finetuned ViT, totally failed) | Yes | Yes | 8 | 224 | No | 61.8 |
| FLAN-T5-XL | LoRA, 8 frames (finetuned ViT briefly for 5 epochs, not too bad) | Yes | Yes | 8 | 224 | No | 71.4 |
| FLAN-T5-XL | LoRA, 8 frames (finetuned ViT, results were not good, i.e., 68.4) | Yes | Yes | 8 | 224 | No | 68.4 |
| FLAN-T5-XL | No LoRA, 8 frame 336x pixel (higher resolution doesn't seem to help, training becomes slower, interpolated embeddings even worse result) | No | No | 8 | 364 | No | 70.7 |
| FLAN-T5-XL | LoRA+SCST | No | No | 16 | 224 | Yes | 79.2 |
| FLAN-T5-XL | LoRA+SCST (changed hyper-parameters) | No | No | 16 | 224 | Yes | 79.1 |
| FLAN-T5-XL | LoRA+SCST (changed hyper-parameters) | No | No | 16 | 224 | Yes | 76.8 |
| FLAN-T5-XL | LoRA+SCST (changed hyper-parameters) | No | No | 16 | 224 | Yes | 79.5 |
| FLAN-T5-XL | LoRA+SCST (changed hyper-parameters) | No | No | 16 | 224 | Yes | 78.7 |
| FLAN-T5-XL | LoRA+SCST (changed hyper-parameters) | No | No | 16 | 224 | Yes | 79.0 |
| FLAN-T5-XL | LoRA+32 frames, no RL (conclusion: 32 frames are not very helpful) | No | No | 32 | 224 | No | 72.2 |
| FLAN-T5-XL | LoRA+SCST+32 frames (conclusion: 32 frames are not very helpful) | No | No | 32 | 224 | Yes | 79.1 |
| FLAN-T5-XL | LoRA+SCST+32 frames (new hyper-parameters) | No | No | 32 | 224 | Yes | 79.3 |
| FLAN-T5-XL | LoRA+32 frames, no RL (seems re-trained once) | No | No | 32 | 224 | No | 72.5 |
| FLAN-T5-XL | LoRA+32 frames, re-trained for RL, seems worse | No | No | 32 | 224 | Yes | 77.8 |
| FLAN-T5-XL | LoRA+32 frames, re-trained for RL, seems ineffective | No | No | 32 | 224 | Yes | 78.3 |
| FLAN-T5-XL | LoRA+32 frames, re-trained for RL, seems ineffective | No | No | 32 | 224 | Yes | 76.2 |
| OPT-2.7b | Changed LLM, feels not as good as FLAN-T5, also very sensitive to hyper-parameters. | - | - | - | - | - | 71.3 |
| OPT-2.7b | Pre-trained BLIP-2 using only 4M data | - | - | - | - | - | B1: 85.0, B2: 71.9, B3: 58.3, B4: 45.6, M: 31.8, R: 64.6, C: 65.7 |
| OPT-6.7b | Switched to 6.7B model but failed 3 times, (i.e., CIDEr: 63.x) | - | - | - | - | - | 63.x |
| Vicuna-7b | Used a larger model, results didn't seem to improve, conclusion: Vicuna-7b is unnecessary, too slow and large | No | No | - | - | No | 72.7 |
| Vicuna-7b | Also tried SCST, it seems Vicuna SCST (RL) hardly improves (e.g., to 79), possibly because this model is already aligned with human preferences, reducing RL efficiency. | Yes | Yes | - | - | Yes | 74.8 |
| Vicuna-7b | Also tried SCST, it seems Vicuna SCST (RL) hardly improves (e.g., to 79), possibly because this model is already aligned with human preferences, reducing RL efficiency. | Yes | Yes | - | - | Yes | 75.8 |
| SimVLG | Also ran a random SimVLG (the simplest example using only LLM), used LLM without image-text pretraining, results were poor, indicating not only LLM is needed but also image-text alignment. | No | No | - | - | No | 57.3 |
