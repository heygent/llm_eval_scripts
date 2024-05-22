from dataclasses import dataclass
import torch
from transformers import BitsAndBytesConfig
from lm_eval.models.huggingface import HFLM
from lm_eval.models.vllm_causallms import VLLM


MODEL_SHORTNAMES = {
    "phi3": "microsoft/Phi-3-mini-4k-instruct",
    "mixtral": "mistralai/Mixtral-8x7B-Instruct-v0.1" 
}


def makemodel(
    pretrained: str,
    flashattention: bool = False,
    vllm: bool = False,
    quantization_4bit: bool = True,
    **kwargs
):
    if pretrained in MODEL_SHORTNAMES:
        pretrained = MODEL_SHORTNAMES[pretrained]
    
    cls = VLLM if vllm else HFLM

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    ) if quantization_4bit else None

    return cls(
        pretrained,
        attn_implementation='flash_attention_2' if flashattention else None,
        quantization_config=quantization_config,
        device_map="auto",
        **kwargs
    )