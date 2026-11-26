"""
    Every model family in one place, for MIHBench inference (src/infer_mihbench.py) and training (src/train.py,
    src/cf_data.py). Both build their inputs with the same Inputs class, so a format change reaches both.
      MODELS               name -> (folder under models/, family); names as in scripts/download_models.py
      <Family>Inputs       text prompts + image paths (+ the text after each prompt: the answer, in training) ->
                           the model inputs, right-padded. It holds only the processor or tokenizer, so DataLoader
                           workers can take it (cf_data.Collator).
      <Family>(model_dir, load_4bit, sdpa)   MIHBench inference with the settings of the family's official script
                           (pgtrece/DAB): .answer(image_paths, prompt) -> text
      <Family>.for_training(model_dir)       (bf16 weights on the CPU, the language model that gets gradient
                           checkpointing, Inputs)
      <Family>.answer_outputs(model, batch, device)   training forward to the answer position (only the hidden
                           states there and one after go through lm_head)
      <Family>.train_mode(model)             training mode, but the frozen vision encoder stays in eval mode
    LORA_TARGETS: the language model's attention and MLP projections; the vision encoder and the projector stay
    frozen. LLaVA-OV runs in fp16 at inference but trains in bf16 (fp16 training without loss scaling is unstable).
    InternVL is remote code: training replaces its <IMG_CONTEXT> embeddings with the tile features, as its
    generate() does, and runs its language model with sdpa instead of the hard-wired eager attention (~10k tokens
    per case).
"""
import sys

import torch
import torchvision.transforms as T
from PIL import Image

from prompts import PROMPTS
from runinfo import pick_device

# language-model layers only: model.language_model.layers.N (Qwen2.5-VL, LLaVA-OV),
# language_model.model.layers.N (InternVL); vision layers never match
LORA_TARGETS = r".*language_model\.(model\.)?layers\.\d+\.(self_attn\.[qkvo]_proj|mlp\.(gate|up|down)_proj)"
NOT_INPUTS = ("labels", "answer_index", "target")  # batch keys the training collator adds


def open_images(paths, size=None):
    images = [Image.open(p).convert("RGB") for p in paths]
    return [im.resize((size, size)) for im in images] if size else images


def internvl_tiles(image, max_num=12, size=448):
    """dynamic_preprocess from the InternVL2.5 model card: cut the image into the grid of at most max_num
    size x size tiles whose aspect ratio is closest to the image's, then add a thumbnail of the whole image."""
    w, h = image.size
    grids = sorted({(i, j) for n in range(1, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1)
                    if i * j <= max_num}, key=lambda g: g[0] * g[1])
    best, best_diff = (1, 1), float("inf")
    for g in grids:
        diff = abs(w / h - g[0] / g[1])
        if diff < best_diff:
            best, best_diff = g, diff
        elif diff == best_diff and w * h > 0.5 * size * size * g[0] * g[1]:
            best = g
    cols, rows = best
    resized = image.resize((size * cols, size * rows))
    tiles = [resized.crop((c * size, r * size, (c + 1) * size, (r + 1) * size))
             for r in range(rows) for c in range(cols)]
    return tiles + [image.resize((size, size))] if len(tiles) > 1 else tiles


def internvl_chat(model):
    """The text InternVL's model.chat() puts around a query, split at the query: (before, after)."""
    conv = sys.modules[type(model).__module__].get_conv_template(model.template)
    conv.system_message = model.system_message
    conv.append_message(conv.roles[0], "<<QUERY>>")
    conv.append_message(conv.roles[1], None)
    return tuple(conv.get_prompt().split("<<QUERY>>"))


# ---- inputs: prompt + images -> model inputs, the same for inference and training ----

class QwenInputs:
    """Qwen2.5-VL: images resized to 672x672 (576 tokens each), the processor's chat template."""
    FAMILY = "qwen2.5-vl"  # its prompt in configs/prompts.yaml

    def __init__(self, processor):
        self.processor, self.tokenizer, self.template = processor, processor.tokenizer, PROMPTS[self.FAMILY]

    def encode(self, prompts, image_paths, texts_after=None):
        texts, images = [], []
        for prompt, paths, after in zip(prompts, image_paths, texts_after or [""] * len(prompts)):
            content = [{"type": "image"} for _ in paths] + [{"type": "text", "text": prompt}]
            text = self.processor.apply_chat_template([{"role": "user", "content": content}],
                                                      tokenize=False, add_generation_prompt=True)
            texts.append(text + after)
            images += open_images(paths, 672)
        return self.processor(text=texts, images=images, padding=True, return_tensors="pt")


class LlavaOVInputs:
    """LLaVA-OV: images resized to 672x672, '<image>\\n' per image, '\\n', the prompt. The images of a case go in
    as one nested list, one multi-image sample: base resolution only (729 + 1 tokens per image). A flat list would
    be treated as separate single images and cut into anyres tiles (~3,700 tokens each)."""
    FAMILY = "llava-ov"

    def __init__(self, processor):
        self.processor, self.tokenizer, self.template = processor, processor.tokenizer, PROMPTS[self.FAMILY]

    def encode(self, prompts, image_paths, texts_after=None):
        texts = [("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
                  + "<image>\n" * len(paths) + "\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n" + after)
                 for prompt, paths, after in zip(prompts, image_paths, texts_after or [""] * len(prompts))]
        images = [open_images(paths, 672) for paths in image_paths]  # nested: one multi-image sample per case
        return self.processor(text=texts, images=images, padding=True, return_tensors="pt")


class InternVLInputs:
    """InternVL2.5: each image (not resized) cut into up to 12 448px tiles plus a thumbnail; the tiles of all
    images as one <image> (num_image_token tokens per tile), then '\\n' and the prompt; chat = the model's chat
    template split around the query (internvl_chat). Adds pixel_values (all tiles of the batch, in order)."""
    FAMILY = "internvl2.5"
    transform = T.Compose([T.Resize((448, 448), interpolation=T.InterpolationMode.BICUBIC), T.ToTensor(),
                           T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    def __init__(self, tokenizer, chat, num_image_token):
        self.tokenizer, self.chat, self.num_image_token = tokenizer, chat, num_image_token
        self.context_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
        self.template = PROMPTS[self.FAMILY]

    def tiles(self, paths):
        """The tiles of the images, in order: (tiles, 3, 448, 448)."""
        return torch.stack([self.transform(t) for im in open_images(paths) for t in internvl_tiles(im)])

    def encode(self, prompts, image_paths, texts_after=None):
        texts, pixels = [], []
        for prompt, paths, after in zip(prompts, image_paths, texts_after or [""] * len(prompts)):
            tiles = self.tiles(paths)
            pixels.append(tiles)
            image = "<img>" + "<IMG_CONTEXT>" * (self.num_image_token * len(tiles)) + "</img>"
            texts.append(self.chat[0] + image + "\n" + prompt + self.chat[1] + after)
        batch = self.tokenizer(texts, padding=True, return_tensors="pt")
        batch["pixel_values"] = torch.cat(pixels)
        return batch


# ---- families: loading, inference and the training forward ----

class VLM:
    """What the families share. A family sets FAMILY (its prompt), Inputs, LEGACY_PROMPT (what runs that did not
    record their prompt used, for infer_mihbench.check_prompt) and VISION (its vision encoder's module path)."""
    VISION = None

    @classmethod
    def answer_outputs(cls, model, batch, device):
        """(logits (cases, 2, vocab) in float32 at the answer position (predicting Yes / No) and the next one
        (predicting <|im_end|>), last hidden state (cases, H) at the answer position). model: the peft model."""
        net = model.get_base_model()
        h = cls.hidden_states(net, batch, device)
        rows = torch.arange(h.shape[0], device=h.device)
        idx = batch["answer_index"].to(h.device)
        at = torch.stack([h[rows, idx], h[rows, idx + 1]], 1)
        return cls.lm_head(net)(at).float(), at[:, 0]

    @staticmethod
    def hidden_states(net, batch, device):
        """Qwen2.5-VL, LLaVA-OV: the processor's outputs are the inputs."""
        inputs = {k: v.to(device, net.dtype) if v.is_floating_point() else v.to(device)
                  for k, v in batch.items() if k not in NOT_INPUTS}
        return net.model(**inputs, use_cache=False).last_hidden_state

    @staticmethod
    def lm_head(net):
        return net.lm_head

    @classmethod
    def train_mode(cls, model):
        """Training mode (LoRA dropout on) except for the frozen vision encoder."""
        model.train()
        try:
            model.get_base_model().get_submodule(cls.VISION).eval()
        except AttributeError:
            pass


class QwenVL(VLM):
    """Qwen2.5-VL/qwen2.5vl_eval.py: images resized to 672x672 (576 tokens each), the question as is,
    greedy, at most 128 new tokens. --load-4bit is the official run: 4-bit weights and eager attention
    (without it: bf16 and sdpa)."""
    FAMILY, Inputs = QwenInputs.FAMILY, QwenInputs
    LEGACY_PROMPT = "{question}"
    VISION = "model.visual"

    def __init__(self, path, load_4bit=False, sdpa=False):
        from transformers import Qwen2_5_VLForConditionalGeneration
        kwargs = {"dtype": torch.bfloat16}
        if load_4bit:  # needs CUDA and `pip install bitsandbytes`
            from transformers import BitsAndBytesConfig
            kwargs.update(quantization_config=BitsAndBytesConfig(load_in_4bit=True), device_map="auto",
                          attn_implementation="eager")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(path, **kwargs).eval()
        if not load_4bit:
            self.model.to(pick_device())
        self.attn = self.model.config._attn_implementation
        self.inputs = self.load_inputs(path)

    @classmethod
    def load_inputs(cls, path):
        from transformers import AutoProcessor
        return cls.Inputs(AutoProcessor.from_pretrained(path, use_fast=False))  # the official run's preprocessing

    @torch.inference_mode()
    def answer(self, image_paths, prompt):
        inputs = self.inputs.encode([prompt], [image_paths]).to(self.model.device)
        out = self.model.generate(**inputs, max_new_tokens=128, do_sample=False, temperature=None)
        return self.inputs.processor.batch_decode(out[:, inputs.input_ids.shape[1]:],
                                                  skip_special_tokens=True)[0].strip()

    @classmethod
    def for_training(cls, path):
        from transformers import Qwen2_5_VLForConditionalGeneration
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(path, dtype=torch.bfloat16)
        return model, model, cls.load_inputs(path)


class LlavaOV(VLM):
    """LLaVA-NeXT-Interleave/llava_next_eval.py (the paper's LLaVA): images resized to 672x672 and each
    encoded once at the base resolution, '<image>\\n' per image then '\\n' + the question, greedy,
    at most 1024 new tokens, fp16."""
    FAMILY, Inputs = LlavaOVInputs.FAMILY, LlavaOVInputs
    LEGACY_PROMPT = "{question}"
    VISION = "model.vision_tower"

    def __init__(self, path, load_4bit=False, sdpa=False):
        from transformers import LlavaOnevisionForConditionalGeneration
        self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(path, dtype=torch.float16).eval()
        self.model.to(pick_device())
        self.attn = self.model.config._attn_implementation
        self.inputs = self.load_inputs(path)

    @classmethod
    def load_inputs(cls, path):
        from transformers import AutoProcessor
        return cls.Inputs(AutoProcessor.from_pretrained(path, use_fast=False))

    @torch.inference_mode()
    def answer(self, image_paths, prompt):
        inputs = self.inputs.encode([prompt], [image_paths])
        inputs = inputs.to(self.model.device, torch.float16)  # casts only the float tensors (pixels)
        out = self.model.generate(**inputs, max_new_tokens=1024, do_sample=False,
                                  pad_token_id=self.inputs.tokenizer.eos_token_id)
        return self.inputs.processor.batch_decode(out[:, inputs.input_ids.shape[1]:],
                                                  skip_special_tokens=True)[0].strip()

    @classmethod
    def for_training(cls, path):
        from transformers import LlavaOnevisionForConditionalGeneration
        model = LlavaOnevisionForConditionalGeneration.from_pretrained(path, dtype=torch.bfloat16)
        return model, model, cls.load_inputs(path)


class InternVL(VLM):
    """InternVL2.5/internvl2.5_eval.py: each image (not resized) cut into up to 12 448px tiles plus a thumbnail,
    the tiles of all images given as one '<image>', then the prompt (the official script always appends
    ' Please answer with yes or no.'), greedy, at most 1024 new tokens. The official run uses eager attention,
    which builds a tokens x tokens matrix per layer and runs out of memory on 24 GB GPUs; sdpa=True computes the
    same attention without it."""
    FAMILY, Inputs = InternVLInputs.FAMILY, InternVLInputs
    LEGACY_PROMPT = "{question} Please answer with yes or no."  # the earlier code added it inside answer()
    VISION = "vision_model"

    def __init__(self, path, load_4bit=False, sdpa=False):
        from transformers import AutoModel, AutoTokenizer
        self.model = AutoModel.from_pretrained(path, dtype=torch.bfloat16, use_flash_attn=False,
                                               trust_remote_code=True).eval().to(pick_device())
        if sdpa:  # the remote code hard-wires eager into the language model's config
            self.model.language_model.config._attn_implementation = "sdpa"
        self.attn = self.model.language_model.config._attn_implementation
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
        self.inputs = InternVLInputs(tokenizer, internvl_chat(self.model), self.model.num_image_token)

    @torch.inference_mode()
    def answer(self, image_paths, prompt):
        pixel_values = self.inputs.tiles(image_paths).to(self.model.device, torch.bfloat16)
        return self.model.chat(self.inputs.tokenizer, pixel_values, "<image>\n" + prompt,
                               dict(max_new_tokens=1024, do_sample=False)).strip()

    @classmethod
    def for_training(cls, path):
        from transformers import AutoModel, AutoTokenizer
        model = AutoModel.from_pretrained(path, dtype=torch.bfloat16, use_flash_attn=False, trust_remote_code=True)
        model.language_model.config._attn_implementation = "sdpa"
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
        inputs = InternVLInputs(tokenizer, internvl_chat(model), model.num_image_token)
        model.img_context_token_id = inputs.context_id
        return model, model.language_model, inputs

    @staticmethod
    def hidden_states(net, batch, device):
        """The tile features go into the <IMG_CONTEXT> embeddings, as InternVL's generate() does."""
        lm, ids = net.language_model, batch["input_ids"].to(device)
        vit = net.extract_feature(batch["pixel_values"].to(device, net.dtype))
        emb = lm.get_input_embeddings()(ids)
        emb = emb.masked_scatter((ids == net.img_context_token_id).unsqueeze(-1).expand_as(emb), vit.to(emb.dtype))
        return lm.model(inputs_embeds=emb, attention_mask=batch["attention_mask"].to(device),
                        use_cache=False).last_hidden_state

    @staticmethod
    def lm_head(net):
        return net.language_model.lm_head


MODELS = {  # name -> (folder under models/, family)
    "qwen2.5-vl-3b": ("Qwen2.5-VL-3B-Instruct", QwenVL),
    "qwen2.5-vl-7b": ("Qwen2.5-VL-7B-Instruct", QwenVL),
    "internvl2.5-4b": ("InternVL2_5-4B", InternVL),
    "llava-ov-7b": ("llava-onevision-qwen2-7b-ov-hf", LlavaOV),
}
