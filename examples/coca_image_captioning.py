import torch
from PIL import Image
from lavis.models import load_model_and_preprocess

# model, vis_processors, _  = load_model_and_preprocess(name="coca", model_type="coca_ViT-L-14")

# raw_image = Image.open("../cat.jpg").convert("RGB")
# image = vis_processors["eval"](raw_image).unsqueeze(0)
# with torch.no_grad():
#     loss = model({"image": image, "text_input": ["a cat sitting on ground"]})
#     generated = model.generate({"image": image, "text_input": None})

# print(loss.loss, generated)


model, vis_processors, _  = load_model_and_preprocess(name="videococa", model_type="coca_ViT-L-14")

raw_image = Image.open("../cat.jpg").convert("RGB")
image = vis_processors["eval"](raw_image).unsqueeze(0).unsqueeze(0) # [1,1,3,224,224]
frames = torch.cat((image, image), dim=1) # [B,T,C,H,W] --> [1,2,3,224,224]
with torch.no_grad():
    loss = model({"image": frames, "text_input": ["a cat sitting on ground"]})
    generated = model.generate({"image": frames, "text_input": None})

print(loss.loss, generated)