from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

from src.config import HF_CACHE_DIR, MODELS_DIR
from src.utils import ensure_directories


DEFAULT_SUMMARIZATION_MODEL = "sshleifer/distilbart-cnn-12-6"
DEFAULT_RESPONSE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def sanitize_model_name(model_name: str) -> str:
    return model_name.replace("/", "--")


def local_model_dir(model_name: str) -> Path:
    return MODELS_DIR / "hf_local" / sanitize_model_name(model_name)


def ensure_local_seq2seq_model(model_name: str) -> Path:
    """Download once, then reuse a local Hugging Face seq2seq model directory."""
    ensure_directories([HF_CACHE_DIR, MODELS_DIR / "hf_local"])
    model_dir = local_model_dir(model_name)
    if model_dir.exists():
        return model_dir

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=HF_CACHE_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=HF_CACHE_DIR)
    tokenizer.save_pretrained(model_dir)
    model.save_pretrained(model_dir)
    return model_dir


def load_local_seq2seq(model_name: str):
    model_dir = ensure_local_seq2seq_model(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir, local_files_only=True)
    model.eval()
    return tokenizer, model


def ensure_local_causal_model(model_name: str) -> Path:
    ensure_directories([HF_CACHE_DIR, MODELS_DIR / "hf_local"])
    model_dir = local_model_dir(model_name)
    if model_dir.exists():
        return model_dir

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=HF_CACHE_DIR)
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=HF_CACHE_DIR)
    tokenizer.save_pretrained(model_dir)
    model.save_pretrained(model_dir)
    return model_dir


def load_local_causal(model_name: str):
    model_dir = ensure_local_causal_model(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True)
    model.eval()
    return tokenizer, model


def generate_text(
    seq2seq_model,
    prompt: str,
    max_new_tokens: int,
    min_new_tokens: int = 0,
) -> str:
    tokenizer, model = seq2seq_model
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    )
    with torch.no_grad():
        generated_ids = model.generate(
            **encoded,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
        )
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()


def generate_chat_response(causal_model, prompt: str, max_new_tokens: int) -> str:
    tokenizer, model = causal_model
    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template"):
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        formatted_prompt = prompt

    encoded = tokenizer(
        formatted_prompt, return_tensors="pt", truncation=True, max_length=2048
    )
    with torch.no_grad():
        generated_ids = model.generate(
            **encoded,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = generated_ids[0][encoded["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
