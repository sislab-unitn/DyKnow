import argparse
import json
import os
from argparse import Namespace

from openai import OpenAI
from tqdm import tqdm

from utils import get_questions

API_KEY = "######### YOUR API KEY ###########################"
CLIENT = OpenAI(api_key=API_KEY)


def configure_subparsers(subparsers) -> None:
    """
    Configure a new subparser.

    Arguments
    ---------
    subparsers: subpparser
        A subparser, where an additional parser will be attached.
    """
    parser = subparsers.add_parser(
        "openai",
        help="Run generation with OpenAI models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "model",
        metavar="MODEL",
        choices={"gpt-3", "chatgpt", "gpt-4"},
        type=str,
        help="OpenAI model to query.",
    )
    parser.add_argument(
        "--prompt",
        metavar="PROMPT",
        type=str,
        default="Answer with the name only.",
        help="Prompt for Q&A generation.",
    )

    parser.set_defaults(func=main)


def questionAnswering(question, model, client, args):
    question = f"{args.prompt}\n\n{question}" if args.use_prompt else question
    if model == "davinci-002":
        output = client.completions.create(model=model, prompt=question)
        response = str(dict(output.choices[0])["text"]).strip()

    else:
        output = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": f"{question}"}],
            temperature=0,
            max_tokens=args.max_length,  # Adjust as needed
            n=1,
        )  # Number of responses to generate
        response = output.choices[0].message.content
    return response


def main(args: Namespace):
    # GPT-3
    if args.model == "gpt-3":
        PATH_TO_CONVERTED_WEIGHTS = "davinci-002"
    # ChatGPT
    elif args.model == "chatgpt":
        PATH_TO_CONVERTED_WEIGHTS = "gpt-3.5-turbo-1106"
    # GPT-4
    elif args.model == "gpt-4":
        PATH_TO_CONVERTED_WEIGHTS = "gpt-4-1106-preview"

    out_path = os.path.join(args.out_dir, PATH_TO_CONVERTED_WEIGHTS.split("/").pop())
    os.makedirs(out_path, exist_ok=True)

    questions = get_questions(args.grc_path)

    for category, category_questions in tqdm(questions.items(), desc="Categories"):
        answers = {}
        for key, elem in tqdm(
            category_questions.items(), desc=f"Generating Answ. for {category}"
        ):
            key_split = key.split("|")
            question_types = elem["question_types"]

            decoded_replies = []
            for q in elem["questions"]:
                reply = questionAnswering(q, PATH_TO_CONVERTED_WEIGHTS, CLIENT, args)
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
            to_modify[entry_key].update(
                {
                    "questions": {
                        t: f"{args.prompt}\n\n{q}" if args.use_prompt else q
                        for (t, q) in zip(question_types, elem["questions"])
                    }
                }
            )

        with open(f"{out_path}/{category}_answers.json", "w") as f:
            json.dump(answers, f, indent=4)
