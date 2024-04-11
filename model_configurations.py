import torch
from transformers import BitsAndBytesConfig
from lm_eval.models.huggingface import HFLM
from lm_eval.models.vllm_causallms import VLLM


def _default_quantization_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )


def mixtral(**kwargs):
    return HFLM(
        pretrained="mistralai/Mixtral-8x7B-Instruct-v0.1",
        quantization_config=_default_quantization_config(),
        device_map="auto",
        **kwargs
    )


def mixtral_flashattention(**kwargs):
    return mixtral(
        attn_implementation="flash_attention_2",
        **kwargs
    )


def mixtral_vllm(**kwargs):
    return VLLM(
        pretrained="mistralai/Mixtral-8x7B-Instruct-v0.1",
        quantization_config=_default_quantization_config(),
        device_map="auto",
        **kwargs
    )
