import argparse
from argparse import Namespace

import torch

from models import (
    gpt2_xl, t5_3b, flan_t5_xl, gpt_j_6b, api, bloom_7b,
    falcon_7b, llama2_7b, vicuna_7b, mistral_7b, mixtral_8x7b
)

def get_args() -> Namespace:
    """
    Parse command line arguments.

    Returns
    -------
    parsed_args: Namespace instance
        Parsed arguments passed through command line.
    """

    parser = argparse.ArgumentParser(
        prog="python -m generate_answers",
        description="Generate answers to questions using a model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # arguments of the parser
    parser.add_argument(
        "--use-prompt",
        action="store_true",
        help="Use the model specific prompt.",
    )
    parser.add_argument(
        "--device",
        metavar="DEVICE",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for generation.",
    )
    parser.add_argument(
        "--max-length",
        metavar="LENGTH",
        type=int,
        default=20,
        help="Max number of tokens that can be generated.",
    )
    parser.add_argument(
        "--out-dir",
        metavar="DIR_NAME",
        type=str,
        default="results",
        help="Destination folder to save the generation results.",
    )
    parser.add_argument(
        "--grc-path",
        metavar="FILE_PATH",
        type=str,
        default="../grc_generated.json",
        help="Path to the file containing Q&A.",
    )
    

    # subparsers
    subparsers = parser.add_subparsers(help="sub-commands help")
    gpt2_xl.configure_subparsers(subparsers)
    t5_3b.configure_subparsers(subparsers)
    flan_t5_xl.configure_subparsers(subparsers)
    gpt_j_6b.configure_subparsers(subparsers)
    api.configure_subparsers(subparsers)
    bloom_7b.configure_subparsers(subparsers)
    falcon_7b.configure_subparsers(subparsers)
    llama2_7b.configure_subparsers(subparsers)
    vicuna_7b.configure_subparsers(subparsers)
    mistral_7b.configure_subparsers(subparsers)
    mixtral_8x7b.configure_subparsers(subparsers)

    # parse arguments
    parsed_args = parser.parse_args()

    if parsed_args.use_prompt:
        parsed_args.out_dir = f"{parsed_args.out_dir}_w_prompt"

    return parsed_args
    

def main():
    args = get_args()
    args.func(args)


if __name__ == "__main__":
    main()