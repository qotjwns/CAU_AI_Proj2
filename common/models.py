"""
    VLMs for the experiments: load a model once, then answer (images, prompt) greedily and keep the logits of
    the first generated token for the answer tokens (Yes, No, yes, no).
      load_model(cfg, answer_tokens, device)   cfg: one entry of `models` in an experiment's config.yaml
      model.answer(image_paths, prompt)        -> {"response", "first_token", "logits": {token: float}}
    Input format, image size and max_new_tokens follow each family's official MIHBench script (pgtrece/DAB).
    The <Family>Inputs classes build the model inputs from the processor or tokenizer alone (no weights).
"""
import importlib.util
import json
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image


def open_images(paths, size=None):
    images = [Image.open(p).convert("RGB") for p in paths]
    return [im.resize((size, size)) for im in images] if size else images


def answer_token_ids(tokenizer, tokens):
    """{token: id}; each answer token must be one token, or its logit would not be the answer's."""
    ids = {}
    for token in tokens:
        encoded = tokenizer.encode(token, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(f"{token!r} is {len(encoded)} tokens for this tokenizer, not 1")
        ids[token] = encoded[0]
    return ids


def decode_answer(tokenizer, new_ids, first_logits, answer_ids):
    """The answer of one call: generated text, its first token, and the raw logits of the answer tokens at the
    first generated position (before repetition penalty or any other logits processor)."""
    first_logits = first_logits.float()
    return {"response": tokenizer.decode(new_ids, skip_special_tokens=True).strip(),
            "first_token": tokenizer.decode(new_ids[:1]),
            "logits": {token: first_logits[i].item() for token, i in answer_ids.items()}}


# ---- Qwen2.5-VL ----

class QwenInputs:
    """Qwen2.5-VL inputs as in Qwen2.5-VL/qwen2.5vl_eval.py: every image resized to image_size x image_size
    (672: 576 visual tokens per image, however many images there are), the images then the text in one user
    turn, the processor's chat template (with its default system prompt)."""

    def __init__(self, path, image_size):
        from transformers import AutoProcessor
        self.processor = AutoProcessor.from_pretrained(path, use_fast=False)  # the official run's preprocessing
        self.tokenizer = self.processor.tokenizer
        self.image_size = image_size

    def text(self, n_images, prompt):
        content = [{"type": "image"} for _ in range(n_images)] + [{"type": "text", "text": prompt}]
        return self.processor.apply_chat_template([{"role": "user", "content": content}],
                                                  tokenize=False, add_generation_prompt=True)

    def encode(self, image_paths, prompt):
        return self.processor(text=[self.text(len(image_paths), prompt)],
                              images=open_images(image_paths, self.image_size), return_tensors="pt")


class QwenVL:
    """Qwen2.5-VL 3B / 7B. Greedy: the model's generation_config samples, so do_sample=False and
    temperature=None override it; its repetition_penalty 1.05 stays, as in the official run."""

    def __init__(self, cfg, answer_tokens, device):
        from transformers import Qwen2_5_VLForConditionalGeneration
        self.inputs = QwenInputs(cfg["path"], cfg["image_size"])
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            cfg["path"], dtype=getattr(torch, cfg["dtype"]), attn_implementation=cfg["attn"]).eval().to(device)
        self.max_new_tokens = cfg["max_new_tokens"]
        self.answer_ids = answer_token_ids(self.inputs.tokenizer, answer_tokens)

    @torch.inference_mode()
    def answer(self, image_paths, prompt):
        batch = self.inputs.encode(image_paths, prompt).to(self.model.device)
        out = self.model.generate(**batch, max_new_tokens=self.max_new_tokens, do_sample=False, temperature=None,
                                  return_dict_in_generate=True, output_logits=True)
        new_ids = out.sequences[0, batch["input_ids"].shape[1]:]
        return decode_answer(self.inputs.tokenizer, new_ids, out.logits[0][0], self.answer_ids)


# ---- InternVL2.5 ----

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


class InternVLInputs:
    """InternVL2.5 inputs as its model.chat() builds them (InternVL2.5/internvl2.5_eval.py calls chat() without
    num_patches_list): each image (not resized) cut into up to max_tiles 448px tiles plus a thumbnail, the tiles
    of all images as one <image> of num_image_token tokens per tile, '<image>\\n' + the prompt in the model's
    conversation template (conversation.py in the model folder, with its own system message)."""
    transform = T.Compose([T.Resize((448, 448), interpolation=T.InterpolationMode.BICUBIC), T.ToTensor(),
                           T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    def __init__(self, path, max_tiles):
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
        config = json.loads((Path(path) / "config.json").read_text())
        image_size = config["force_image_size"] or config["vision_config"]["image_size"]
        self.num_image_token = int((image_size // config["vision_config"]["patch_size"]) ** 2
                                   * config["downsample_ratio"] ** 2)
        spec = importlib.util.spec_from_file_location("internvl_conversation", Path(path) / "conversation.py")
        conversation = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(conversation)
        self.template = conversation.get_conv_template(config["template"])
        self.system_message = self.template.system_message  # what InternVLChatModel.__init__ takes too
        self.sep = self.template.sep.strip()  # <|im_end|>: chat() stops there
        self.eos_id = self.tokenizer.convert_tokens_to_ids(self.sep)
        self.context_id = self.tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
        self.max_tiles = max_tiles

    def query(self, prompt, n_tiles):
        template = self.template.copy()
        template.append_message(template.roles[0], "<image>\n" + prompt)
        template.append_message(template.roles[1], None)
        image = "<img>" + "<IMG_CONTEXT>" * (self.num_image_token * n_tiles) + "</img>"
        return template.get_prompt().replace("<image>", image, 1)

    def encode(self, image_paths, prompt):
        tiles = [t for im in open_images(image_paths) for t in internvl_tiles(im, self.max_tiles)]
        batch = self.tokenizer(self.query(prompt, len(tiles)), return_tensors="pt")
        batch["pixel_values"] = torch.stack([self.transform(t) for t in tiles])
        return batch


class InternVL:
    """InternVL2.5-4B (remote code). Generates as model.chat() does (greedy, stops at <|im_end|>), but through
    model.generate() so the first-step logits come back. attn sets the language model's attention: the remote
    code hard-wires eager, which runs out of memory on 24 GB GPUs; sdpa computes the same attention."""

    def __init__(self, cfg, answer_tokens, device):
        from transformers import AutoModel
        self.inputs = InternVLInputs(cfg["path"], cfg["max_tiles"])
        self.model = AutoModel.from_pretrained(cfg["path"], dtype=getattr(torch, cfg["dtype"]), use_flash_attn=False,
                                               trust_remote_code=True).eval().to(device)
        self.model.language_model.config._attn_implementation = cfg["attn"]
        self.model.img_context_token_id = self.inputs.context_id  # chat() sets it before generate()
        if (self.model.num_image_token, self.model.system_message) != (self.inputs.num_image_token,
                                                                       self.inputs.system_message):
            raise RuntimeError("InternVLInputs disagrees with the loaded model on num_image_token or system message")
        self.max_new_tokens = cfg["max_new_tokens"]
        self.answer_ids = answer_token_ids(self.inputs.tokenizer, answer_tokens)

    @torch.inference_mode()
    def answer(self, image_paths, prompt):
        batch = self.inputs.encode(image_paths, prompt)
        device, dtype = self.model.device, self.model.dtype
        out = self.model.generate(pixel_values=batch["pixel_values"].to(device, dtype),
                                  input_ids=batch["input_ids"].to(device),
                                  attention_mask=batch["attention_mask"].to(device),
                                  max_new_tokens=self.max_new_tokens, do_sample=False, eos_token_id=self.inputs.eos_id,
                                  return_dict_in_generate=True, output_logits=True)
        new_ids = out.sequences[0]  # generated from input embeddings: the prompt is not in the output
        answer = decode_answer(self.inputs.tokenizer, new_ids, out.logits[0][0], self.answer_ids)
        answer["response"] = answer["response"].split(self.inputs.sep)[0].strip()  # as chat() does
        return answer


# ---- LLaVA-OneVision ----

class LlavaOVInputs:
    """LLaVA-OV inputs as in LLaVA-NeXT-Interleave/llava_next_eval.py: images resized to image_size x image_size,
    '<image>\\n' per image, '\\n', the prompt. Several images go in as one multi-image sample, which LLaVA-OV
    encodes at the base resolution (729 + 1 newline = 730 tokens per image). A single image would be cut into
    anyres tiles (~3,700 tokens); with anyres False it gets the multi-image encoding instead (see encode)."""

    def __init__(self, path, image_size, anyres):
        from transformers import AutoProcessor
        self.processor = AutoProcessor.from_pretrained(path, use_fast=False)
        self.tokenizer = self.processor.tokenizer
        self.image_size, self.anyres = image_size, anyres
        self.image_tokens = self.processor.num_image_tokens + 1  # one image of a multi-image sample, + newline
        if self.processor.vision_feature_select_strategy == "default":
            self.image_tokens -= 1

    def text(self, n_images, prompt):
        return ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
                + "<image>\n" * n_images + "\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n")

    def encode(self, image_paths, prompt):
        from transformers import BatchFeature
        images, text = open_images(image_paths, self.image_size), self.text(len(image_paths), prompt)
        if len(images) > 1 or self.anyres:
            return self.processor(text=[text], images=[images], return_tensors="pt")  # nested: one sample
        # One image at the base resolution, as one image of a multi-image sample: the pixels of a two-copy
        # sample cut to the first copy, the image token repeated as many times as such an image takes, and
        # batch_num_images 2 so that the model does not expect anyres tiles for it (it only reads n != 1).
        pair = self.processor.image_processor([images * 2], return_tensors="pt")
        batch = self.tokenizer([text.replace("<image>", "<image>" * self.image_tokens)], return_tensors="pt")
        return BatchFeature({**batch, "pixel_values": pair["pixel_values"][:1], "image_sizes": pair["image_sizes"][:1],
                             "batch_num_images": torch.tensor([2])})


class LlavaOV:
    """LLaVA-OneVision 7B in fp16 (the official script's precision), greedy."""

    def __init__(self, cfg, answer_tokens, device):
        from transformers import LlavaOnevisionForConditionalGeneration
        self.inputs = LlavaOVInputs(cfg["path"], cfg["image_size"], cfg["anyres"])
        self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            cfg["path"], dtype=getattr(torch, cfg["dtype"]), attn_implementation=cfg["attn"]).eval().to(device)
        self.max_new_tokens = cfg["max_new_tokens"]
        self.answer_ids = answer_token_ids(self.inputs.tokenizer, answer_tokens)

    @torch.inference_mode()
    def answer(self, image_paths, prompt):
        batch = self.inputs.encode(image_paths, prompt).to(self.model.device, self.model.dtype)  # floats only
        out = self.model.generate(**batch, max_new_tokens=self.max_new_tokens, do_sample=False,
                                  pad_token_id=self.inputs.tokenizer.eos_token_id,
                                  return_dict_in_generate=True, output_logits=True)
        new_ids = out.sequences[0, batch["input_ids"].shape[1]:]
        return decode_answer(self.inputs.tokenizer, new_ids, out.logits[0][0], self.answer_ids)


FAMILIES = {"qwen2.5-vl": QwenVL, "internvl2.5": InternVL, "llava-ov": LlavaOV}


def load_model(cfg, answer_tokens, device):
    family = cfg["family"]
    if family not in FAMILIES:
        raise SystemExit(f"{family}: unknown model family (known: {', '.join(FAMILIES)})")
    return FAMILIES[family](cfg, answer_tokens, device)
