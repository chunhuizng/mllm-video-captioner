import random
import logging

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn

from .blip2 import Blip2Base, disabled_train
from .modeling_llama import LlamaForCausalLM
from transformers import LlamaTokenizer, LlamaConfig

from lavis.common.registry import registry ### added
from lavis.models.base_model import BaseModel

from transformers import StoppingCriteria, StoppingCriteriaList
import os

logger = logging.getLogger(__name__)


class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops = [], encounters=1):
        super().__init__()
        self.stops = stops
        self.ENCOUNTERS = encounters

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        stop_count = 0
        for stop in self.stops:
            stop_count = (stop == input_ids[0]).sum().item()
        if stop_count >= self.ENCOUNTERS:
            return True
        return False


@registry.register_model("videochat_pt")
class VideoChat_pt(Blip2Base):
    """
    VideoChat model.
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "videochat_vicuna7b": "configs/models/videochat/videochat_7b.yaml",
        "videochat_vicuna13b": "configs/models/videochat/videochat_13b.yaml",
    }

    def __init__(self, 
        vit_model="eva_clip_g",
        vit_model_path=None,
        q_former_model_path=None,
        llama_model_path=None,
        videochat_model_path="",  
        freeze_vit=True,
        freeze_qformer=True,
        # vit
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        low_resource=False, # use 8 bit and put vit in cpu
        # uniformerv2
        freeze_mhra=False,
        temporal_downsample=True,
        no_lmhra=False,
        double_lmhra=False,
        lmhra_reduction=2.0,
        gmhra_layers=8,
        gmhra_drop_path_rate=0.,
        gmhra_dropout=0.5,
        # qformer
        num_query_token=32,
        qformer_hidden_dropout_prob=0.1,
        qformer_attention_probs_dropout_prob=0.1,
        qformer_drop_path_rate=0.1,
        extra_num_query_token=32,
        # prompt
        prompt_path="",
        img_prompt_path="",
        prompt_template="",
        max_txt_len=32,
        end_sym='\n',
        # debug
        debug=False,
        ):
        super().__init__()

        self.tokenizer = self.init_tokenizer()
        self.low_resource = low_resource

        self.vit_precision = vit_precision
        logger.info(f'Loading VIT. Use fp16: {vit_precision}')
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, 
            use_grad_checkpoint, vit_precision, vit_model_path,
            temporal_downsample=temporal_downsample,
            no_lmhra=no_lmhra, 
            double_lmhra=double_lmhra,
            lmhra_reduction=lmhra_reduction, 
            gmhra_layers=gmhra_layers, 
            gmhra_drop_path_rate=gmhra_drop_path_rate,
            gmhra_dropout=gmhra_dropout, 
        )
        if freeze_vit:
            logger.info("freeze vision encoder")
            if not freeze_mhra:
                open_list = []
                for name, param in self.visual_encoder.named_parameters():
                    if 'mhra' not in name:
                        param.requires_grad = False
                    else:
                        open_list.append(name)
                logger.info(f"open module: {open_list}")
                logger.info("open ln_vision")
            else:
                for name, param in self.visual_encoder.named_parameters():
                    param.requires_grad = False
                self.visual_encoder = self.visual_encoder.eval()
                self.visual_encoder.train = disabled_train
                for name, param in self.ln_vision.named_parameters():
                    param.requires_grad = False
                self.ln_vision = self.ln_vision.eval()
                self.ln_vision.train = disabled_train
        logger.info('Loading VIT Done')

        logger.info('Loading Q-Former')
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features,
            qformer_hidden_dropout_prob=qformer_hidden_dropout_prob,
            qformer_attention_probs_dropout_prob=qformer_attention_probs_dropout_prob,
            qformer_drop_path_rate=qformer_drop_path_rate,
        )
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
        self.load_from_pretrained(model_path=q_former_model_path)
        logger.info(f"Add extra {extra_num_query_token} tokens in QFormer")
        self.extra_query_tokens = nn.Parameter(
            torch.zeros(1, extra_num_query_token, self.query_tokens.shape[-1])
        )

        if freeze_qformer:
            logger.info("freeze Qformer")
            for name, param in self.Qformer.named_parameters():
                param.requires_grad = False
            self.Qformer = self.Qformer.eval()
            self.Qformer.train = disabled_train
            self.query_tokens.requires_grad = False
        logger.info('Loading Q-Former Done')

        logger.info('Loading LLAMA')
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(llama_model_path, use_fast=False, cache_dir="/home/anonymous/new_ssd/cache_dir")
        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token

        if self.low_resource:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model_path,
                torch_dtype=torch.float16,
                load_in_8bit=True,
                device_map="auto",
                use_cache=True,
                cache_dir="/home/anonymous/new_ssd/cache_dir"
            )
        else:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model_path,
                torch_dtype=torch.float16,
                use_cache=True,
                cache_dir="/home/anonymous/new_ssd/cache_dir"
            )

        logger.info("freeze LLAMA")
        for name, param in self.llama_model.named_parameters():
            param.requires_grad = False
        logger.info('Loading LLAMA Done')

        self.llama_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.llama_model.config.hidden_size
        )
        self.max_txt_len = max_txt_len
        self.end_sym = end_sym

        if prompt_path:
            # self.prompt_list = self.process_prompt(prompt_path, prompt_template)
            self.prompt_list = [
                "<Video><VideoHere></Video> Describe the following video concisely.",
                "<Video><VideoHere></Video> Provide a brief description of the given video clip.",
                "<Video><VideoHere></Video> Offer a succinct explanation of the footage presented.",
                "<Video><VideoHere></Video> Summarize the visual content of the following video.",
                "<Video><VideoHere></Video> Give a short and clear explanation of the subsequent video clip.",
                "<Video><VideoHere></Video> Share a concise interpretation of the video provided.",
                "<Video><VideoHere></Video> Present a compact description of the clip's key features.",
                "<Video><VideoHere></Video> Relay a brief, clear account of the video shown.",
                "<Video><VideoHere></Video> Render a clear and concise summary of the video below.",
                "<Video><VideoHere></Video> Write a terse but informative summary of the following video clip.",
                "<Video><VideoHere></Video> Create a compact narrative representing the video presented.",
            ]
        else:
            self.prompt_list = []
        if img_prompt_path:
            # self.img_prompt_list = self.process_prompt(img_prompt_path, prompt_template)
            self.img_prompt_list = [
                "<Image><ImageHere></Image> Describe the following image concisely.",
                "<Image><ImageHere></Image> Provide a brief description of the given image.",
                "<Image><ImageHere></Image> Offer a succinct explanation of the picture presented.",
                "<Image><ImageHere></Image> Summarize the visual content of the following image.",
                "<Image><ImageHere></Image> Give a short and clear explanation of the subsequent image.",
                "<Image><ImageHere></Image> Share a concise interpretation of the image provided.",
                "<Image><ImageHere></Image> Present a compact description of the photo's key features.",
                "<Image><ImageHere></Image> Relay a brief, clear account of the picture shown.",
                "<Image><ImageHere></Image> Render a clear and concise summary of the photo below.",
                "<Image><ImageHere></Image> Write a terse but informative summary of the following picture.",
                "<Image><ImageHere></Image> Create a compact narrative representing the image presented.",
            ]
        else:
            self.img_prompt_list = []

        # load weights of VideoChat
        if videochat_model_path:
            logger.info(f"Load VideoChat from: {videochat_model_path}")
            ckpt = torch.load(videochat_model_path, map_location="cpu")
            msg = self.load_state_dict(ckpt['model'], strict=False)
            logger.info(msg)

    def process_prompt(self, prompt_path, prompt_template):
        with open(prompt_path, 'r') as f:
            raw_prompts = f.read().splitlines()
        filted_prompts = [raw_prompt for raw_prompt in raw_prompts]
        prompt_list = [prompt_template.format(p) for p in filted_prompts]
        logger.info(f'Load {len(prompt_list)} training prompts')
        logger.info(f'Prompt: {prompt_list}')
        return prompt_list

    def vit_to_cpu(self):
        self.ln_vision.to("cpu")
        self.ln_vision.float()
        self.visual_encoder.to("cpu")
        self.visual_encoder.float()

    def encode_img(self, image):
        device = image.device
        if self.low_resource:
            self.vit_to_cpu()
            image = image.to("cpu")

        with self.maybe_autocast():
            T = image.shape[1]
            use_image = True if T == 1 else False
            # image = image.permute(0, 2, 1, 3, 4) # [B,T,C,H,W] -> [B,C,T,H,W]

            image_embeds = self.ln_vision(self.visual_encoder(image)).to(device)
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)

            query_tokens = torch.cat([self.query_tokens, self.extra_query_tokens], dim=1)
            query_tokens = query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            inputs_llama = self.llama_proj(query_output.last_hidden_state)
            atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(image.device)
        return inputs_llama, atts_llama, use_image

    def prompt_wrap(self, img_embeds, atts_img, prompt, use_image=False):
        if prompt:
            batch_size = img_embeds.shape[0]
            if use_image:
                p_before, p_after = prompt.split('<ImageHere>')
            else:
                p_before, p_after = prompt.split('<VideoHere>')
            p_before_tokens = self.llama_tokenizer(
                p_before, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
            p_after_tokens = self.llama_tokenizer(
                p_after, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
            p_before_embeds = self.llama_model.model.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1)
            p_after_embeds = self.llama_model.model.embed_tokens(p_after_tokens.input_ids).expand(batch_size, -1, -1)
            wrapped_img_embeds = torch.cat([p_before_embeds, img_embeds, p_after_embeds], dim=1)
            wrapped_atts_img = atts_img[:, :1].expand(-1, wrapped_img_embeds.shape[1])
            return wrapped_img_embeds, wrapped_atts_img
        else:
            return img_embeds, atts_img

    def forward(self, samples):
        image = samples["image"]
        text_input = samples["text_input"]
        img_embeds, atts_img, use_image = self.encode_img(image)
        if self.prompt_list:
            if use_image:
                prompt = random.choice(self.img_prompt_list)
            else:
                prompt = random.choice(self.prompt_list)
            img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img, prompt, use_image)

        self.llama_tokenizer.padding_side = "right"
        text = [t + self.end_sym for t in text_input]

        to_regress_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False
        ).to(image.device)

        targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
        )

        empty_targets = (
            torch.ones([atts_img.shape[0], atts_img.shape[1]+1],
                    dtype=torch.long).to(image.device).fill_(-100)  # plus one for bos
        )
        targets = torch.cat([empty_targets, targets], dim=1)

        batch_size = img_embeds.shape[0]
        bos = torch.ones([batch_size, 1],
                         dtype=to_regress_tokens.input_ids.dtype,
                         device=to_regress_tokens.input_ids.device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.llama_model.model.embed_tokens(bos)
        atts_bos = atts_img[:, :1]

        to_regress_embeds = self.llama_model.model.embed_tokens(to_regress_tokens.input_ids)
        inputs_embeds = torch.cat([bos_embeds, img_embeds, to_regress_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, atts_img, to_regress_tokens.attention_mask], dim=1)

        with self.maybe_autocast():
            outputs = self.llama_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )
    
        return dict(
            loss=outputs.loss,
        )

    @torch.no_grad()
    def generate(
        self, 
        samples,
        use_nucleus_sampling=False,
        num_beams=5,
        max_length=30,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=1.0,
        num_captions=1,
        temperature=1,
    ):
        image = samples["image"]
        img_embeds, atts_img, use_image = self.encode_img(image)
        if self.prompt_list:
            if use_image:
                prompt = random.choice(self.img_prompt_list)
            else:
                prompt = random.choice(self.prompt_list)
            img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img, prompt, use_image)

        batch_size = img_embeds.shape[0]
        bos = torch.ones([batch_size, 1],
                         dtype=torch.long,
                         device=img_embeds.device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.llama_model.model.embed_tokens(bos)
        atts_bos = atts_img[:, :1]

        with self.maybe_autocast():
            inputs_embeds = torch.cat([bos_embeds, img_embeds], dim=1)
            attention_mask = torch.cat([atts_bos, atts_img], dim=1)

            stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=[2277, 29937], encounters=1)]) # '###' can be encoded in two different ways
            outputs = self.llama_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_length=max_length,
                min_length=min_length,
                # eos_token_id=self.eos_token_id,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
                stopping_criteria=stopping_criteria
            )

        output_text = self.llama_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        output_text = [text.replace("#", "").strip() for text in output_text]

        return output_text

    @classmethod
    def from_config(cls, config):
        # pretrained_path
        vit_model = config.get("vit_model", "eva_clip_g")
        vit_model_path = config.get("vit_model_path", None)
        q_former_model_path = config.get("q_former_model_path", None)
        llama_model_path = config.get("llama_model_path")
        videochat_model_path = config.get("videochat_model_path", "")  
        freeze_vit = config.get("freeze_vit", True)
        freeze_qformer = config.get("freeze_qformer", True)
        # vit
        img_size = config.get("img_size")
        drop_path_rate = config.get("drop_path_rate", 0)
        use_grad_checkpoint = config.get("use_grad_checkpoint", False)
        vit_precision = config.get("vit_precision", "fp16")
        low_resource = config.get("low_resource", False) # use 8 bit and put vit in cpu
        # uniformerv2
        freeze_mhra = config.get("freeze_mhra", False)
        temporal_downsample = config.get("temporal_downsample", True)
        no_lmhra = config.get("no_lmhra", False)
        double_lmhra = config.get("double_lmhra", False)
        lmhra_reduction = config.get("lmhra_reduction", 2.0)
        gmhra_layers = config.get("gmhra_layers", 8)
        gmhra_drop_path_rate = config.get("gmhra_drop_path_rate", 0.)
        gmhra_dropout = config.get("gmhra_dropout", 0.5)
        # qformer
        num_query_token = config.get("num_query_token")
        qformer_hidden_dropout_prob = config.get("qformer_hidden_dropout_prob", 0.1)
        qformer_attention_probs_dropout_prob = config.get("qformer_attention_probs_dropout_prob", 0.1)
        qformer_drop_path_rate = config.get("qformer_drop_path_rate", 0.1)
        extra_num_query_token = config.get("extra_num_query_token", 32)
        # prompt
        prompt_path = config.get("prompt_path", "")
        img_prompt_path = config.get("img_prompt_path", "")
        prompt_template = config.get("prompt_template", "")
        max_txt_len = config.get("max_txt_len", 32)
        end_sym = config.get("end_sym", '\n')
        # debug
        debug = config.get("debug", False)

        model = cls(
            vit_model=vit_model,
            vit_model_path=vit_model_path,
            q_former_model_path=q_former_model_path,
            llama_model_path=llama_model_path,
            videochat_model_path=videochat_model_path,  
            freeze_vit=freeze_vit,
            freeze_qformer=freeze_qformer,
            # vit
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            low_resource=low_resource, # use 8 bit and put vit in cpu
            # uniformerv2
            freeze_mhra=freeze_mhra,
            temporal_downsample=temporal_downsample,
            no_lmhra=no_lmhra,
            double_lmhra=double_lmhra,
            lmhra_reduction=lmhra_reduction,
            gmhra_layers=gmhra_layers,
            gmhra_drop_path_rate=gmhra_drop_path_rate,
            gmhra_dropout=gmhra_dropout,
            # qformer
            num_query_token=num_query_token,
            qformer_hidden_dropout_prob=qformer_hidden_dropout_prob,
            qformer_attention_probs_dropout_prob=qformer_attention_probs_dropout_prob,
            qformer_drop_path_rate=qformer_drop_path_rate,
            extra_num_query_token=extra_num_query_token,
            # prompt
            prompt_path=prompt_path,
            img_prompt_path=img_prompt_path,
            prompt_template=prompt_template,
            max_txt_len=max_txt_len,
            end_sym=end_sym,
            # debug
            debug=False,
        )
        # model.load_checkpoint_from_config(config)  ### do not load model now

        # load the pretrained/finetuned models for evaluation
        load_pretrained = config.get("load_pretrained", False)
        pretrained_filename = config.get("pretrained", "")
        if load_pretrained:
            if os.path.isfile(pretrained_filename):
                checkpoint = torch.load(pretrained_filename, map_location="cpu")
            else:
                raise RuntimeError("checkpoint url or path is invalid")

            msg = model.load_state_dict(checkpoint["model"], strict=False)
            logging.info("load checkpoint from %s" % pretrained_filename)

        return model