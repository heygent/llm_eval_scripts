import lm_eval
import wandb
from lm_eval.logging_utils import WandbLogger

from model_configurations import mixtral


def main():
    wandb.login()
    evalmodel = mixtral()

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

    results = lm_eval.simple_evaluate(
        model=evalmodel,
        tasks=tasks,
        log_samples=True
    )

    wandb_logger = WandbLogger(
        project="gentiletti-llm",
        job_type="eval"
    )

    wandb_logger.post_init(results)
    wandb_logger.log_eval_results()
    wandb_logger.log_eval_samples(results["samples"])  # if log_samples


if __name__ == "__main__":
    main()
