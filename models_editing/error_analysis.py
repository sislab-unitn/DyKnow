import argparse
import gc
import json
import math
import os
import random
from argparse import Namespace

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from EasyEdit.easyeditor import (
    BaseEditor, ROMEHyperParams, MEMITHyperParams, SERACHparams, 
)
from utils import generate_sample_answers


def get_args() -> Namespace:
    """
    Parse command line arguments.

    Returns
    -------
    parsed_args: Namespace instance
        Parsed arguments passed through command line.
    """

    parser = argparse.ArgumentParser(
        prog="python -m error_analysis",
        description="Perform error analysis incrementaly for a specific editing method.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # arguments of the parser
    parser.add_argument(
        "edit_alg",
        metavar="EDIT_ALG_NAME",
        type=str,
        choices={"ROME", "MEMIT", "SERAC"},
        help="Editing algorithm to update a model.",
    )
    parser.add_argument(
        "hparams_path",
        metavar="HPARAMS_PATH",
        type=str,
        help="Path to the hparams path with the configuration.",
    )
    parser.add_argument(
        "edit_dataset_path",
        metavar="DATASET_PATH",
        type=str,
        help="Path to the editing dataset.",
    )
    parser.add_argument(
        "--splits",
        metavar="SPLIT_PERCENTAGE",
        nargs="+",
        type=float,
        default=[.25, .50, .75, 1],
        help="Max number of tokens that can be generated.",
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
        default="error_analysis",
        help="Destination folder to save the generation results.",
    )
    parser.add_argument(
        "--batch-editing",
        action="store_true",
        help="If set, perform batch editing (if the method allows it).",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        help="If set, saves the edited model.",
    )
    parser.add_argument(
        "--test-generation",
        action="store_true",
        help="If set, test model generation before and after editing.",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="If set, parallelize on available gpus.",
    )

    # parse arguments
    parsed_args = parser.parse_args()

    if parsed_args.edit_alg == "ROME":
        parsed_args.edit_alg = ROMEHyperParams
    elif parsed_args.edit_alg == "MEMIT":
        parsed_args.edit_alg = MEMITHyperParams
    elif parsed_args.edit_alg == "SERAC":
        parsed_args.edit_alg = SERACHparams
    else:
        raise NotImplementedError

    return parsed_args

def main(args: Namespace):

    # Loading config
    hparams = args.edit_alg.from_hparams(args.hparams_path)
    hparams.model_parallel = args.parallel
    
    # Dataset
    with open(args.edit_dataset_path, "r") as f:
        dataset = json.load(f)
    
    answers = {}
    ## edit descriptor: prompt that you want to edit
    prompts = []
    ## subject: suject in the prompt (for ROME and MEMIT)
    subjects = []
    ## edit target: expected output
    targets = []

    # Initialize the tokenizer for the dataset
    tokenizer = AutoTokenizer.from_pretrained(hparams.model_name)

    # set the seed and shuffle the dataset only once at the beginning
    random.seed(42)
    random.shuffle(dataset)
    for sample in dataset:
        prompt = sample["prompt"] 
        subject = sample["subject"] 
        target = sample["target"]

        if "Mistral-7B-Instruct" in hparams.model_name:
            prompt = f"{tokenizer.bos_token}[INST] Answer with the name only. {prompt} [/INST]"
            target = f"{target}{tokenizer.eos_token}"
        elif "Llama-2-7b-chat-hf" in hparams.model_name:
            prompt = f"{tokenizer.bos_token}[INST] <<SYS>>\nAnswer with the name only.\n<</SYS>>\n\n{prompt} [/INST]"
            target = f"{target}{tokenizer.eos_token}"

        prompts.append(prompt)
        subjects.append(subject)
        targets.append(target)
    

    for split in args.splits:
        # Editor
        editor = BaseEditor.from_hparams(hparams)
        # Tokenizer
        tokenizer = editor.tok

        if args.batch_editing:
            hparams.batch_size = len(subjects[:math.ceil(split*len(dataset))])
            _, edited_model, _ = editor.batch_edit(
                prompts=prompts[:math.ceil(split*len(dataset))],
                subject=subjects[:math.ceil(split*len(dataset))],
                target_new=targets[:math.ceil(split*len(dataset))],
                keep_original_weight=False,
                test_generation=args.test_generation,
            )
        else:
            _, edited_model, _ = editor.edit(
                prompts=prompts[:math.ceil(split*len(dataset))],
                subject=subjects[:math.ceil(split*len(dataset))],
                target_new=targets[:math.ceil(split*len(dataset))],
                keep_original_weight=False,
                test_generation=args.test_generation,
            )
            

        train_ds = None
        qa_to_update = {}
        for sample in tqdm(dataset[:math.ceil(split*len(dataset))], desc="Generating"):
            domain = sample["domain"]
            element = sample["element"]
            attribute = sample["attribute"]

            if domain not in qa_to_update:
                qa_to_update[domain] = {}
            if element not in qa_to_update[domain]:
                qa_to_update[domain][element] = {}
            if attribute is None:
                qa_to_update[domain][element] = {
                    "quesions": sample["questions"],
                    "answers": sample["answers"]
                }
            else:
                if attribute not in qa_to_update[domain][element]:
                    qa_to_update[domain][element][attribute] = {
                        "questions": sample["questions"],
                        "answers": sample["answers"]
                    }

            generate_sample_answers(
                sample,
                answers,
                editor,
                train_ds,
                tokenizer,
                edited_model,
                f"cuda:{hparams.device}",
                args.max_length,
                hparams,
            )

        out_dir = os.path.join(
            args.out_dir,
            hparams.model_name.split('/')[-1],
            hparams.alg_name, 
            "batch_editing" if args.batch_editing else "single_edits",
            str(split)
        )
        os.makedirs(out_dir, exist_ok=True)

        for domain in answers:
            with open(os.path.join(out_dir, f"{domain}_answers.json"), "w") as f:
                json.dump(answers[domain], f, indent=4)

        with open(os.path.join(out_dir, f"qa_to_update.json"), "w") as f:
            json.dump(qa_to_update, f, indent=4)

        del edited_model
        del editor
        del tokenizer
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    args = get_args()
    main(args)