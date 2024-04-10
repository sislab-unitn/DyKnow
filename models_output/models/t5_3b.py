import argparse
import json
import os
from argparse import Namespace

from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration

from utils import get_questions

PATH_TO_CONVERTED_WEIGHTS = "t5-3b"

def configure_subparsers(subparsers) -> None:
    """
    Configure a new subparser.

    Arguments
    ---------
    subparsers: subpparser
        A subparser, where an additional parser will be attached.
    """
    parser = subparsers.add_parser(
        "t5-3b",
        help="Run generation with T5-3B.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--prompt",
        metavar="PROMPT",
        type=str,
        default="Answer with the name only:",
        help="Prompt for Q&A generation.",
    )

    parser.set_defaults(func=main)
    

def main(args: Namespace):
    out_path = os.path.join(args.out_dir, PATH_TO_CONVERTED_WEIGHTS.split('/').pop())
    os.makedirs(out_path, exist_ok=True)

    questions = get_questions(args.grc_path)

    model = T5ForConditionalGeneration.from_pretrained(PATH_TO_CONVERTED_WEIGHTS).to(args.device)
    tokenizer = T5Tokenizer.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)

    for category, category_questions in tqdm(questions.items(), desc="Categories"):
        answers = {}
        for key, elem in tqdm(category_questions.items(), desc=f"Generating Answ. for {category}"):
            key_split = key.split("|")
            question_types = elem["question_types"]

            decoded_replies = []
            for q in elem["questions"]:
                if args.use_prompt:
                    inputs = tokenizer(f"{args.prompt} {q}", return_tensors='pt', padding=True)
                else:
                    inputs = tokenizer(q, return_tensors='pt', padding=True)
                inputs = inputs.to(args.device)
                generate_ids = model.generate(
                    inputs.input_ids,
                    max_new_tokens=args.max_length,
                )
                reply = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
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
            to_modify[entry_key].update({
                "questions": {
                    t: f"{args.prompt} {q}" if args.use_prompt else q 
                    for (t, q) in zip(question_types, elem["questions"])
                }
            })

        with open(f"{out_path}/{category}_answers.json", "w") as f:
            json.dump(answers, f, indent=4)