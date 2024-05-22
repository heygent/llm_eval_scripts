import click
import lm_eval
from lm_eval.models.huggingface import HFLM
from lm_eval.models.vllm_causallms import VLLM
from lm_eval.logging_utils import WandbLogger
import torch
from transformers import BitsAndBytesConfig
import wandb

from model_configurations import makemodel

MODEL_SHORTNAMES = {
    "phi3": "microsoft/Phi-3-mini-4k-instruct",
    "mixtral": "mistralai/Mixtral-8x7B-Instruct-v0.1" 
}

DEFAULT_TASKS = [
    # "ai2_arc",

    # gsm8k
    # "math_word_problems",
    # "chain_of_thought",
    # "self_consistency",

    "hellaswag",
    # "mmlu",
    # "truthfulqa_mc1",
    # "winogrande",
]


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


def run_lm_eval(model, tasks=DEFAULT_TASKS):
    return lm_eval.simple_evaluate(
        model=model,
        tasks=tasks,
        log_samples=True
    )

def wandb_log_run(results):
    wandb.login()
    wandb_logger = WandbLogger(
        project="gentiletti-llm",
        job_type="eval"
    )
    wandb_logger.post_init(results)
    wandb_logger.log_eval_results()
    wandb_logger.log_eval_samples(results["samples"])  # if log_samples


@click.command()
@click.argument('model', type=click.Choice(['phi3', 'mixtral'], case_sensitive=False))
@click.option('--flash-attention', is_flag=True, default=False)
@click.option('--vllm', is_flag=True, default=False)
def main(model, flashattention: bool, vllm: bool):
    click.echo(f"Running {model} with {flashattention=} and {vllm=}")
    model = makemodel('phi3', flashattention=True, vllm=True)
    results = run_lm_eval(model)
    wandb_log_run(results)

if __name__ == "__main__":
    main()