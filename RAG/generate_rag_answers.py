import argparse
import json
import os
from argparse import Namespace
from typing import Dict, List, Optional

import torch

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BatchEncoding


def get_args() -> Namespace:
    """
    Parse command line arguments.

    Returns
    -------
    parsed_args: Namespace instance
        Parsed arguments passed through command line.
    """

    parser = argparse.ArgumentParser(
        prog="python -m generate_rag_answers",
        description="Generate answers to questions using RAG with a specific model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )


    parser.add_argument(
        "model_name",
        metavar="MODEL_NAME",
        choices={"mistral", "llama2"},
        type=str,
        help="Model name.",
    )
    parser.add_argument(
        "outdated_qa_file",
        metavar="OUTDATED_QA_FILE",
        type=str,
        help="Path to the QA file containing model-specific outdated questions.",
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
        "--passages-path",
        metavar="FILE_PATH",
        type=str,
        default="editing_passages.json",
        help="Path to the file containing the passages collected from Wikipedia.",
    )
    
    # parse arguments
    parsed_args = parser.parse_args()

    return parsed_args
    

def encode_inputs(
        q: str,
        context: str,
        tokenizer: AutoTokenizer,
        model_name: str,
    ) -> BatchEncoding:

    if "mistral" in model_name:
        # https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1
        messages = [
            {"role": "user", "content": f"CONTEXT: {context}\n\nUsing the CONTEXT above, Answer with the name only: {q}"}
        ]
        inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", padding=True)
        inputs = BatchEncoding({
            'input_ids': inputs,
            'attention_mask': torch.ones_like(inputs),
        })
    elif "llama2" in model_name:
        inputs = tokenizer(
            f"[INST] <<SYS>>\nCONTEXT: {context}\n\nUsing the CONTEXT above, Answer with the name only.\n<</SYS>>\n\n{q} [/INST]", 
            return_tensors='pt', padding=True
            )
    else:
        inputs = tokenizer(q, return_tensors='pt', padding=True)
    
    return inputs


def generate_answers(
        questions: List[str],
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        args: Namespace,
        context: str,
    ) -> Dict[str, Dict[str, str]]:

    res = {
        "questions": {},
        "answers": {}
    }
    for qt, q in questions.items():
        inputs = encode_inputs(q, context, tokenizer, args.model_name)
        res["questions"][qt] = tokenizer.batch_decode(
            inputs.input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        inputs = inputs.to(args.device)
        generate_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_length,
            pad_token_id=tokenizer.eos_token_id,
        )

        res["answers"][qt] = tokenizer.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

    return res


def main():
    args = get_args()

    if args.model_name == "mistral":
        model_path = "mistralai/Mistral-7B-Instruct-v0.1"
    elif args.model_name == "llama2":
        model_path = "meta-llama/Llama-2-7b-chat-hf"
    else:
        raise NotImplementedError
    
    out_dir = os.path.join(args.out_dir, model_path.split('/').pop())
    os.makedirs(out_dir, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Tok config
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    with open(args.outdated_qa_file, "r") as f:
        outdated_questions = json.load(f)

    with open(args.passages_path, "r") as f:
        passages = json.load(f)

    answers = {}
    for domain in outdated_questions:
        if domain not in answers:
            answers[domain] = {}
        for element in tqdm(outdated_questions[domain], desc=domain):
            if element not in answers[domain]:
                answers[domain][element] = {}
            if domain in ["countries_byGDP", "organizations"]:
                for attribute in outdated_questions[domain][element]:
                    if attribute not in answers[domain][element]:
                        answers[domain][element][attribute] = {}
                    questions = outdated_questions[domain][element][attribute]["questions"]
                    question_passages = passages[domain][element][attribute]

                    matches = []
                    for matches_per_category in question_passages["matches"].values():
                        matches += [m["paragraph"]["text"] for m in matches_per_category]

                    assert len(matches) == 1, f"You should have only 1 passage for each question but you have {len(matches)} for {domain} -- {element} -- {attribute}"
                    context = matches.pop()

                    answers[domain][element][attribute] = generate_answers(
                        questions, model, tokenizer, args, context
                    )
            else:
                questions = outdated_questions[domain][element]["questions"]
                question_passages = passages[domain][element]

                matches = []
                for matches_per_category in question_passages["matches"].values():
                    matches += [m["paragraph"]["text"] for m in matches_per_category]

                assert len(matches) == 1, f"You should have only 1 passage for each question but you have {len(matches)} for {domain} -- {element}"
                context = matches.pop()

                answers[domain][element] = generate_answers(
                    questions, model, tokenizer, args, context
                )

    for domain in answers:
        with open(os.path.join(out_dir,  f"{domain}_answers.json"), "w") as f:
            json.dump(answers[domain], f, indent=4)


if __name__ == "__main__":
    main()