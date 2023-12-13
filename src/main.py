from pathlib import Path
import argparse

from train import train_baselines, train_llms
from infer import make_inference

def getArgs():
    parser = argparse.ArgumentParser(description="Parse arguments from command input.")
    parser.add_argument(
        "-t",
        "--task",
        action="store",
        required=True,
        type=str,
        choices=["train", "infer"],
        help='Enter "train" To train models. Enter "infer" to make an inference. ',
    )
    parser.add_argument(
        "-mt",
        "--model_type",
        action="store",
        type=str,
        default="baseline",
        choices=['LLM', 'baseline'],
        help="Choose model type to train.",
    )
    return parser.parse_args()

if __name__ == "__main__":

    Path("outputs/csv").mkdir(parents=True, exist_ok=True)
    Path("outputs/png").mkdir(parents=True, exist_ok=True)
    Path("outputs/scores").mkdir(parents=True, exist_ok=True)
    Path("outputs/model").mkdir(parents=True, exist_ok=True)


    arg = getArgs()

    if arg.task == "train":
        if arg.model_type == "baseline":
            print("training baseline models ...")
            train_baselines(seed=333)
        elif arg.model_type == "LLM":
            print("train LLM model")
            train_llms(seed=123, train_size=0.8)

    elif arg.task == "infer":      
        test_input = "This is a sample. Warning: Your DAI balance is running low. Replenish your Aave account to continue accessing the benefits of decentralized lending and borrowing."
        if arg.model_type == "baseline":
            pred = make_inference(test_input).best_baseline()
        elif arg.model_type == "LLM":
            pred = make_inference(test_input).best_llm()
        print(pred)