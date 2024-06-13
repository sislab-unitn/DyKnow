import argparse
import json
import os
from argparse import Namespace

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils import get_questions


def configure_subparsers(subparsers) -> None:
    """
    Configure a new subparser.

    Arguments
    ---------
    subparsers: subpparser
        A subparser, where an additional parser will be attached.
    """
    parser = subparsers.add_parser(
        "llama3-8b",
        help="Run generation with Llama-3-8B.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--prompt",
        metavar="PROMPT",
        type=str,
        default="Answer with the name only.",
        help="Prompt for Q&A generation.",
    )
    parser.add_argument(
        "--instruct",
        action="store_true",
        help="Use the instruct version of Llama-3.",
    )

    parser.set_defaults(func=main)


def main(args: Namespace):
    # Mistral Instruct
    if args.instruct:
        PATH_TO_CONVERTED_WEIGHTS = "meta-llama/Meta-Llama-3-8B-Instruct"
    else:
        # Mistral
        PATH_TO_CONVERTED_WEIGHTS = "meta-llama/Meta-Llama-3-8B"

    out_path = os.path.join(args.out_dir, PATH_TO_CONVERTED_WEIGHTS.split("/").pop())
    os.makedirs(out_path, exist_ok=True)

    questions = get_questions(args.grc_path)

    model = AutoModelForCausalLM.from_pretrained(
        PATH_TO_CONVERTED_WEIGHTS, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)

    # Tok config
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]

    for category, category_questions in tqdm(questions.items(), desc="Categories"):
        answers = {}
        for key, elem in tqdm(
            category_questions.items(), desc=f"Generating Answ. for {category}"
        ):
            key_split = key.split("|")
            question_types = elem["question_types"]

            decoded_replies = []
            for q in elem["questions"]:
                if args.instruct:
                    inputs = tokenizer(
                        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{args.prompt if args.use_prompt else ''}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{q}<|eot_id|><|start_header_id|>assistant<|end_header_id|>",
                        return_tensors="pt",
                    )
                else:
                    inputs = tokenizer(
                        f"{args.prompt} {q}" if args.use_prompt else q,
                        return_tensors="pt",
                    )
                inputs = inputs.to(args.device)

                generate_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_length,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=terminators,
                )

                reply = tokenizer.batch_decode(
                    generate_ids,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )[0]
                decoded_replies.append(reply)

            if len(key_split) == 1:
                to_modify = answers
                entry_key = key_split[0]
            elif len(key_split) == 2:
                if key_split[0] not in answers:
                    answers[key_split[0]] = {}
                to_modify = answers[key_split[0]]
                entry_key = key_split[1]

            # Add the answers
            to_modify[entry_key] = {
                "answers": {qt: r for (qt, r) in zip(question_types, decoded_replies)}
            }
            # Add the questions
            if args.instruct:
                to_modify[entry_key].update(
                    {
                        "questions": {
                            t: f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{args.prompt if args.use_prompt else ''}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{q}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
                            for (t, q) in zip(question_types, elem["questions"])
                        }
                    }
                )
            else:
                to_modify[entry_key].update(
                    {
                        "questions": {
                            t: f"{args.prompt} {q}" if args.use_prompt else q
                            for (t, q) in zip(question_types, elem["questions"])
                        }
                    }
                )

        with open(f"{out_path}/{category}_answers.json", "w") as f:
            json.dump(answers, f, indent=4)
