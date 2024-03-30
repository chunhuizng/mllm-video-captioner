"""
 Copyright (c) 2023, anonymous.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging

import torch
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from transformers import T5TokenizerFast

from lavis.common.registry import registry
from lavis.models.blip2_models.blip2 import Blip2Base, disabled_train
from lavis.models.blip2_models.modeling_t5 import T5Config, T5ForConditionalGeneration

import itertools
import numpy as np

from lavis.tasks.pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from lavis.tasks.pycocoevalcap.cider.cider import Cider

from peft import LoraConfig, TaskType, get_peft_model


def tokenize(refs, cands, no_op=False):
    # no_op is a debug option to see how significantly not using the PTB tokenizer
    # affects things
    tokenizer = PTBTokenizer()

    if no_op:
        refs = {idx: [r for r in c_refs] for idx, c_refs in enumerate(refs)}
        cands = {idx: [c] for idx, c in enumerate(cands)}

    else:
        refs = {idx: [{'caption':r} for r in c_refs] for idx, c_refs in enumerate(refs)}
        cands = {idx: [{'caption':c}] for idx, c in enumerate(cands)}

        refs = tokenizer.tokenize(refs)
        cands = tokenizer.tokenize(cands)

    return refs, cands


@registry.register_model("blip2_t5")
class Blip2T5(Blip2Base):
    """
    BLIP2 T5 model.
    Supported model types:
        - pretrain_flant5xl: pretrained model with FlanT5-XL
        - pretrain_flant5xl_vitL: pretrained model with FlanT5-XL
        - pretrain_flant5xxl: pretrained model with FlanT5-XXL
        - caption_coco_flant5xl: fintuned image captioning model with FlanT5-XL
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2_t5", "pretrain_flant5xl")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_flant5xl": "configs/models/blip2/blip2_pretrain_flant5xl.yaml",
        "pretrain_flant5xl_vitL": "configs/models/blip2/blip2_pretrain_flant5xl_vitL.yaml",
        "pretrain_flant5xxl": "configs/models/blip2/blip2_pretrain_flant5xxl.yaml",
        "caption_coco_flant5xl": "configs/models/blip2/blip2_caption_flant5xl.yaml",
    }

    def __init__(
        self,
        vit_model="eva_clip_g",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        num_query_token=32,
        t5_model="google/flan-t5-xl",
        prompt="",
        max_txt_len=32,
        apply_lemmatizer=False,
        scst=False,
        beam_size=5,
        lora=False,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
    ):
        """
        apply_lemmatizer: when set to True, postprocess predict_answers() result with lemmas.
        """
        super().__init__()

        self.tokenizer = self.init_tokenizer()

        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logging.info("freeze vision encoder")

        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features
        )
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        self.t5_tokenizer = T5TokenizerFast.from_pretrained(t5_model)   ### cache_dir="/home/anonymous/new_ssd/cache_dir"
        t5_config = T5Config.from_pretrained(t5_model)
        t5_config.dense_act_fn = "gelu"
        self.t5_model = T5ForConditionalGeneration.from_pretrained(
            t5_model, config=t5_config,   ### cache_dir="/home/anonymous/new_ssd/cache_dir"
        )

        for name, param in self.t5_model.named_parameters():
            param.requires_grad = False
            param.data = param.data.bfloat16()

        peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            inference_mode=False, 
            r=lora_r, 
            lora_alpha=lora_alpha, 
            lora_dropout=lora_dropout,
            target_modules=['q', 'v']
        )

        if lora:
            self.t5_model = get_peft_model(self.t5_model, peft_config)
            self.t5_model.print_trainable_parameters()

        self.t5_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.t5_model.config.hidden_size
        )

        self.max_txt_len = max_txt_len
        self.prompt = prompt

        self._apply_lemmatizer = apply_lemmatizer
        self._lemmatizer = None
        self.scst = scst
        self.beam_size = beam_size


    def forward(self, samples):
        if not self.scst: # xe training
            image = samples["image"]

            B, C, T, H, W = image.shape
            image = image.permute(0,2,1,3,4).contiguous() ### B C T H W --> B T C H W
            image = image.reshape(B*T,C,H,W).contiguous() ### B T C H W --> B*T C H W

            with self.maybe_autocast():
                image_embeds = self.ln_vision(self.visual_encoder(image))
                image_embeds = image_embeds.reshape(B, -1, image_embeds.shape[-1]) ### B*T h*w D --> B T*h*w D
                image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
                    image.device
                )

            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            inputs_t5 = self.t5_proj(query_output.last_hidden_state)
            atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(image.device)

            ### new code starts
            text = samples["text_input"]
            samples["text_input"] = [self.prompt] * B
            samples["text_output"] = text
            ### new code ends

            with self.maybe_autocast(dtype=torch.bfloat16):
                input_tokens = self.t5_tokenizer(
                    samples["text_input"],
                    padding="longest",
                    truncation=True,
                    max_length=self.max_txt_len,
                    return_tensors="pt",
                ).to(image.device)
                output_tokens = self.t5_tokenizer(
                    samples["text_output"],
                    padding="longest",
                    truncation=True,
                    max_length=self.max_txt_len,
                    return_tensors="pt",
                ).to(image.device)

                encoder_atts = torch.cat([atts_t5, input_tokens.attention_mask], dim=1)

                targets = output_tokens.input_ids.masked_fill(
                    output_tokens.input_ids == self.t5_tokenizer.pad_token_id, -100
                )

                inputs_embeds = self.t5_model.encoder.embed_tokens(input_tokens.input_ids)
                inputs_embeds = torch.cat([inputs_t5, inputs_embeds], dim=1)

                outputs = self.t5_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=encoder_atts,
                    decoder_attention_mask=output_tokens.attention_mask,
                    return_dict=True,
                    labels=targets,
                )
                loss = outputs.loss

                return {"loss": loss}
            
        else: # scst training: https://arxiv.org/abs/1612.00563
            image = samples["image"]
            B, C, T, H, W = image.shape
            image = image.permute(0,2,1,3,4).contiguous() ### B C T H W --> B T C H W
            image = image.reshape(B*T,C,H,W).contiguous() ### B T C H W --> B*T C H W

            with self.maybe_autocast():
                image_embeds = self.ln_vision(self.visual_encoder(image))
                image_embeds = image_embeds.reshape(B, -1, image_embeds.shape[-1]) ### B*T h*w D --> B T*h*w D

                image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
                    image.device
                )

                query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
                query_output = self.Qformer.bert(
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                )

                inputs_t5 = self.t5_proj(query_output.last_hidden_state)
                atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(image.device)

                ### new code starts
                text = samples["text_input"]
                samples["text_input"] = [self.prompt] * B
                samples["text_output"] = text
                ### new code ends

            with self.maybe_autocast(dtype=torch.bfloat16):
                input_tokens = self.t5_tokenizer(
                    samples["text_input"],
                    padding="longest",
                    truncation=True,
                    max_length=self.max_txt_len,
                    return_tensors="pt",
                ).to(image.device)

                encoder_atts = torch.cat([atts_t5, input_tokens.attention_mask], dim=1)

                inputs_embeds = self.t5_model.encoder.embed_tokens(input_tokens.input_ids)
                inputs_embeds = torch.cat([inputs_t5, inputs_embeds], dim=1)

                outputs = self.t5_model.generate(
                        inputs_embeds=inputs_embeds, 
                        attention_mask=encoder_atts,
                        do_sample=False,
                        top_p=0.9,
                        temperature=1,
                        num_beams=self.beam_size,
                        max_length=32,
                        # min_length=5, ### RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation
                        # eos_token_id=self.eos_token_id, ### AttributeError
                        repetition_penalty=1.0,
                        length_penalty=1.0,
                        num_return_sequences=self.beam_size, ### num_beams
                        return_dict_in_generate=True,
                        output_scores=True,
                    )

                transition_scores = self.t5_model.compute_transition_scores(
                    outputs.sequences, outputs.scores, outputs.beam_indices, normalize_logits=False
                )

                output_length = torch.sum(transition_scores < 0, dim=1)
                sequences_scores = transition_scores.sum(dim=1) / (output_length**1.0)
                sequences_scores = sequences_scores.view(B, -1) # [batch, num_beams]

                caps_gen = self.t5_tokenizer.batch_decode(
                    outputs.sequences, skip_special_tokens=True
                )
                caps_gen = [text.strip() for text in caps_gen]
                caps_gt = list(itertools.chain(*([c, ] * self.beam_size for c in samples["text_output"])))
                caps_gt = [[c] for c in caps_gt]

                caps_gen, caps_gt = tokenize(caps_gt, caps_gen)
                reward = Cider().compute_score(caps_gt, caps_gen)[1].astype(np.float32)
                reward = torch.from_numpy(reward).to(image.device).view(B, self.beam_size)
                reward_baseline = torch.mean(reward, -1, keepdim=True)
                loss = - (sequences_scores) * (reward-reward_baseline)
                loss = loss.mean()

                return {"loss": loss}


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
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
            use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each image.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        image = samples["image"]
        B, C, T, H, W = image.shape
        image = image.permute(0,2,1,3,4).contiguous() ### B C T H W --> B T C H W
        image = image.reshape(B*T,C,H,W).contiguous() ### B T C H W --> B*T C H W

        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image))
            image_embeds = image_embeds.reshape(B, -1, image_embeds.shape[-1]) ### B*T h*w D --> B T*h*w D
 
        image_embeds = image_embeds.float()
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        inputs_t5 = self.t5_proj(query_output.last_hidden_state)
        atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(image.device)

        if "prompt" in samples.keys():
            prompt = samples["prompt"]
        else:
            prompt = self.prompt

        if isinstance(prompt, str):
            prompt = [prompt] * B
        else:
            assert len(prompt) == B, "The number of prompts must be equal to the batch size."

        input_tokens = self.t5_tokenizer(
            prompt, padding="longest", return_tensors="pt"
        ).to(image.device)

        encoder_atts = torch.cat([atts_t5, input_tokens.attention_mask], dim=1)

        with self.maybe_autocast(dtype=torch.bfloat16):
            inputs_embeds = self.t5_model.encoder.embed_tokens(input_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_t5, inputs_embeds], dim=1)

            outputs = self.t5_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=encoder_atts,
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_new_tokens=max_length,
                min_length=min_length,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
            )
            output_text = self.t5_tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )

        return output_text

    def predict_answers(
        self,
        samples,
        num_beams=5,
        inference_method="generate",
        max_len=10,
        min_len=1,
        num_ans_candidates=128,
        answer_list=None,
        prompt="",
        length_penalty=-1,
        **kwargs
    ):
        image = samples["image"]
        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image))
        image_embeds = image_embeds.float()
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        inputs_t5 = self.t5_proj(query_output.last_hidden_state)
        atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(image.device)

        if isinstance(samples["text_input"], str):
            samples["text_input"] = [samples["text_input"]]
        if prompt:
            text_input = [prompt.format(question) for question in samples["text_input"]]
        else:
            text_input = samples["text_input"]

        input_tokens = self.t5_tokenizer(
            text_input, padding="longest", return_tensors="pt"
        ).to(image.device)

        encoder_atts = torch.cat([atts_t5, input_tokens.attention_mask], dim=1)

        with self.maybe_autocast(dtype=torch.bfloat16):
            inputs_embeds = self.t5_model.encoder.embed_tokens(input_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_t5, inputs_embeds], dim=1)

            outputs = self.t5_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=encoder_atts,
                do_sample=False,
                num_beams=num_beams,
                max_new_tokens=max_len,
                min_length=min_len,
                length_penalty=length_penalty,
            )
            output_text = self.t5_tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )

        if self._apply_lemmatizer:
            output_text = self._lemmatize(output_text)

        return output_text

    def _lemmatize(self, answers):
        def apply(answer):
            doc = self.lemmatizer(answer)

            words = []
            for token in doc:
                if token.pos_ in ["NOUN", "VERB"]:
                    words.append(token.lemma_)
                else:
                    words.append(token.text)
            answer = " ".join(words)

            return answer

        return [apply(answer) for answer in answers]

    @property
    def lemmatizer(self):
        if self._lemmatizer is None:
            try:
                import spacy

                self._lemmatizer = spacy.load("en_core_web_sm")
            except ImportError:
                logging.error(
                    """
                    Please install spacy and en_core_web_sm model to apply lemmatization.
                    python -m spacy download en_core_web_sm
                    OR
                    import spacy.cli
                    spacy.cli.download("en_core_web_sm")
                    """
                )
                exit(1)

        return self._lemmatizer

    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        t5_model = cfg.get("t5_model")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)

        prompt = cfg.get("prompt", "")
        max_txt_len = cfg.get("max_txt_len", 32)

        apply_lemmatizer = cfg.get("apply_lemmatizer", False)
        scst = cfg.get("scst", False)

        lora=cfg.get("lora", False)
        lora_r=cfg.get("lora_r", 16)
        lora_alpha=cfg.get("lora_alpha", 32)
        lora_dropout=cfg.get("lora_dropout", 0.05)

        model = cls(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            t5_model=t5_model,
            prompt=prompt,
            max_txt_len=max_txt_len,
            apply_lemmatizer=apply_lemmatizer,
            scst=scst,
            lora=lora,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        model.load_checkpoint_from_config(cfg)

        return model
