"""
    The VLMs for training: loading, which layers get LoRA, and the forward pass up to the answer position.
    Each model family differs here; everything else in training is shared (src/train.py).
      load_model      bf16 weights, the family's collator (src/cf_data.py); InternVL's language model runs with sdpa
                      instead of its hard-wired eager attention (~10k tokens per case)
      LORA_TARGETS    the language model's attention and MLP projections; the vision encoder and the projector stay
                      frozen, and train_mode keeps the vision encoder in eval mode
      answer_outputs  only the hidden states at the answer position and the one after it go through lm_head.
                      InternVL is remote code: its <IMG_CONTEXT> embeddings are replaced by the tile features here,
                      as its generate() does
    LLaVA-OV runs in fp16 at inference but trains in bf16: fp16 training without loss scaling is unstable.
"""
import sys

import torch
from transformers import (AutoModel, AutoProcessor, AutoTokenizer, LlavaOnevisionForConditionalGeneration,
                          Qwen2_5_VLForConditionalGeneration)

from cf_data import InternVLCollator, LlavaOVCollator, QwenCollator

# language-model layers only: model.language_model.layers.N (Qwen2.5-VL, LLaVA-OV),
# language_model.model.layers.N (InternVL); vision layers never match
LORA_TARGETS = r".*language_model\.(model\.)?layers\.\d+\.(self_attn\.[qkvo]_proj|mlp\.(gate|up|down)_proj)"
VISION = ("model.visual", "model.vision_tower", "vision_model")  # Qwen2.5-VL, LLaVA-OV, InternVL
NOT_INPUTS = ("labels", "answer_index", "target")


def internvl_chat(model):
    """The text InternVL's model.chat() puts around a query, split at the query: (before, after)."""
    conv = sys.modules[type(model).__module__].get_conv_template(model.template)
    conv.system_message = model.system_message
    conv.append_message(conv.roles[0], "<<QUERY>>")
    conv.append_message(conv.roles[1], None)
    return tuple(conv.get_prompt().split("<<QUERY>>"))


def load_model(name, model_dir):
    """(model, its language model, collator), bf16 on the CPU. Gradient checkpointing goes on the second."""
    if name.startswith("qwen"):
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_dir, dtype=torch.bfloat16)
        return model, model, QwenCollator(AutoProcessor.from_pretrained(model_dir, use_fast=False))
    if name.startswith("llava"):
        model = LlavaOnevisionForConditionalGeneration.from_pretrained(model_dir, dtype=torch.bfloat16)
        return model, model, LlavaOVCollator(AutoProcessor.from_pretrained(model_dir, use_fast=False))
    model = AutoModel.from_pretrained(model_dir, dtype=torch.bfloat16, use_flash_attn=False, trust_remote_code=True)
    model.language_model.config._attn_implementation = "sdpa"
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, use_fast=False)
    collate = InternVLCollator(tokenizer, internvl_chat(model), model.num_image_token)
    model.img_context_token_id = collate.context_id
    return model, model.language_model, collate


def train_mode(model):
    """Training mode (LoRA dropout on) except for the frozen vision encoder."""
    model.train()
    net = model.get_base_model()
    for path in VISION:
        try:
            net.get_submodule(path).eval()
        except AttributeError:
            pass


def lm_head_of(net):
    return net.language_model.lm_head if hasattr(net, "extract_feature") else net.lm_head  # InternVL, others


def answer_outputs(model, batch, device):
    """(logits (cases, 2, vocab) in float32 at the answer position (predicting Yes / No) and the next one
    (predicting <|im_end|>), last hidden state (cases, H) at the answer position)."""
    net = model.get_base_model()
    if hasattr(net, "extract_feature"):  # InternVL: tile features into the <IMG_CONTEXT> embeddings
        lm, ids = net.language_model, batch["input_ids"].to(device)
        vit = net.extract_feature(batch["pixel_values"].to(device, net.dtype))
        emb = lm.get_input_embeddings()(ids)
        emb = emb.masked_scatter((ids == net.img_context_token_id).unsqueeze(-1).expand_as(emb), vit.to(emb.dtype))
        h = lm.model(inputs_embeds=emb, attention_mask=batch["attention_mask"].to(device),
                     use_cache=False).last_hidden_state
    else:  # Qwen2.5-VL, LLaVA-OV: the processor's outputs are the inputs
        inputs = {k: v.to(device, net.dtype) if v.is_floating_point() else v.to(device)
                  for k, v in batch.items() if k not in NOT_INPUTS}
        h = net.model(**inputs, use_cache=False).last_hidden_state
    rows = torch.arange(h.shape[0], device=h.device)
    idx = batch["answer_index"].to(h.device)
    at = torch.stack([h[rows, idx], h[rows, idx + 1]], 1)
    return lm_head_of(net)(at).float(), at[:, 0]
