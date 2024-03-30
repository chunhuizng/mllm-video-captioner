import torch
from PIL import Image
from lavis.models import load_model_and_preprocess


model, vis_processors, _  = load_model_and_preprocess(name="video_llama", model_type="vicuna_7b")

frames = torch.randn(1,2,3,224,224) # [B,T,C,H,W] --> [1,2,3,224,224]
frames = frames.permute(0,2,1,3,4) # [B,T,C,H,W] --> [B,C,T,H,W]
with torch.no_grad():
    # loss = model({"image": frames, "text_input": ["a cat sitting on ground"]})
    generated = model.generate({"image": frames, "text_input": None})

print(generated)
