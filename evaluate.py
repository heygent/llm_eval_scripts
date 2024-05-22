import lm_eval
import wandb
from lm_eval.logging_utils import WandbLogger

import click

from model_configurations import makemodel


def run_lm_eval(model):
    tasks = [
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