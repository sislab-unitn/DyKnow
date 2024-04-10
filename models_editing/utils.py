import random
from typing import Dict, List, Optional

import torch
from transformers import BatchEncoding, AutoTokenizer

from EasyEdit.easyeditor import apply_ike_to_model, BaseEditor, ZsreDataset

EXCEPTIONS = {
    "athletes_byPayment": {
        "Lionel Messi": "Argentina national association football team",
        "Neymar Jr.": "Brazil national football team",
        "Kylian Mbappé": "France national association football team",
        "Mohamed Salah": "Egypt national football team",
        "Sadio Mané": "Senegal national association football team",
        "Kevin De Bruyne": "Belgium national football team",
        "Harry Kane": "England national association football team",
    }
}

def is_exception(
    answer_name: str,
    category: str,
    element: str,
    attribute: Optional[str],
    exceptions: dict,
) -> bool:
    if category in exceptions:
        if element in exceptions[category]:
            if attribute is None:
                return answer_name in exceptions[category][element]
            else:
                if attribute in exceptions[category][element]:
                    return answer_name in exceptions[category][element][attribute]

    return False


def extract_target(
    answers: List[str],
    exceptions: dict,
    category: str,
    element: str,
    attribute: Optional[str] = None,
) -> str:

    no_end_entries = []
    # Sort the list so that we consider only the latest entry for a given candidate
    for answer in sorted(answers):
        # Skip the answer if it is an exception

        split_answer = answer.split("|")
        name, span = split_answer[0], split_answer[1:]
        name = name.strip()  # remove first and last spaces

        if is_exception(name, category, element, attribute, exceptions):
            continue

        assert len(span) <= 2, "Additional elements in the answer span"

        if len(span) == 1:
            single_span = span.pop()
            if single_span.startswith("S:"):
                no_end_entries.append(name)

    assert (
        len(no_end_entries) == 1
    ), f"There are {len(no_end_entries)} entries with no end for {category} {element} {attribute if attribute else ''}: {no_end_entries}"

    return no_end_entries.pop() 


def prepare_targets(category: str, original: dict, exceptions: dict = EXCEPTIONS) -> dict:
    targets = {}

    if category in ["countries_byGDP", "organizations"]:
        for element, attributes in original[category].items():
            if element not in targets:
                targets[element] = {}
            for attribute, grc_elem in attributes.items():
                target = extract_target(
                    grc_elem["answers"], exceptions, category, element, attribute
                )
                targets[element][attribute] = target
    else:
        for element, grc_elem in original[category].items():
            target = extract_target(
                grc_elem["answers"], exceptions, category, element
            )
            targets[element] = target

    return targets


def return_ike_examples(
    sample: dict,
    q: str,
    editor: BaseEditor,
    train_ds: ZsreDataset,
    realistic: bool,
):
    # in a real case scenario we only have access to the user question
    prompt = q if realistic else sample["prompt"]
    target = '' if realistic else sample["target"]
    requests = editor._prepare_requests([prompt], [target], ground_truth=['<|endoftext|>'], rephrase_prompts=None,
                                        locality_prompts=None, locality_ground_truth=None)

    icl_examples = editor.apply_algo(
        editor.model,
        editor.tok,
        requests.pop(), # since it is a single request
        editor.hparams,
        copy=False,
        return_orig_weights=True,
        keep_original_weight=False,
        train_ds=train_ds
    )

    if realistic:
        # remove the last example since it contains a pair with no target
        icl_examples.pop()

    return icl_examples


def encode_inputs(
        q: str,
        qt: str,
        res: Dict[str, Dict[str, str]],
        tokenizer: AutoTokenizer,
        hparams,
        sample: dict,
        update_questions: bool = False,
        icl_examples: Optional[List[str]] = None,
        error_analysis: bool = False,
        realistic: bool = False,
    ) -> BatchEncoding:

    if error_analysis:
        original_icl_example = icl_examples[-1 if realistic else -2]
        fact_a, prompt_a = icl_examples[-1 if realistic else -2].split('Prompt: ')
        if realistic:
            # take the example with the highest similarity
            new_fact = icl_examples[0].split('Prompt: ')[0]#"New Fact: Not Available\n\n"
            # use it for the user question
            fact_b, prompt_b = f'{new_fact}Prompt: {q}'.split('Prompt: ')
        else:
            fact_b, prompt_b = f'New Fact: {sample["prompt"]} {sample["target"]}\nPrompt: {q}'.split('Prompt: ')
        icl_examples[-1 if realistic else -2] = f"{fact_b}Prompt: {prompt_a}"

    if "Mistral-7B-Instruct" in hparams.model_name:
        # https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1
        if icl_examples is not None and hparams.alg_name == "IKE":
            messages = []
            for icl_e in icl_examples:
                fact, prompt = icl_e.split('Prompt: ')
                messages.append(
                    {"role": "user", "content": f"Answer with the name only. {fact}"}
                )
                messages.append(
                    {"role": "assistant", "content": f" Prompt: {prompt}"}
                )

            inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", padding=True)
            if error_analysis:
                last_pair = f" [INST] Answer with the name only. {fact_a} [/INST] Prompt: {prompt_b}" # we added new fact at the beginning
            elif realistic:
                # take the example with the highest similarity
                new_fact = icl_examples[0].split('Prompt: ')[0]#"New Fact: Not Available\n\n"
                # use it for the user question
                last_pair = f" [INST] Answer with the name only. {new_fact} [/INST] Prompt: {q}"
            else:
                last_pair = f" [INST] Answer with the name only. New Fact: {sample['prompt']} {sample['target']}\n [/INST] Prompt: {q}"
            
            last_turn = tokenizer(
                last_pair,
                return_tensors='pt',
                padding=True,
                add_special_tokens=False
            ).input_ids
            inputs = torch.cat((inputs.squeeze()[:-1], last_turn.squeeze()), 0).unsqueeze(0)
        else:
            messages = [
                {"role": "user", "content": f"Answer with the name only. {q}"}
            ]
            inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", padding=True)
        inputs = BatchEncoding({
            'input_ids': inputs,
            'attention_mask': torch.ones_like(inputs),
        })
    elif "Llama-2-7b-chat-hf" in hparams.model_name:
        if icl_examples is not None and hparams.alg_name == "IKE":
            messages = [f"[INST] <<SYS>>\nAnswer with the name only.\n<</SYS>>\n\n"]
            for i, icl_e in enumerate(icl_examples):
                fact, prompt = icl_e.split('Prompt: ')
                if i == 0:
                    messages.append(
                        f"{fact} [/INST] Prompt: {prompt} {tokenizer.eos_token}{tokenizer.bos_token}"
                    )
                else:
                    messages.append(
                        f"[INST] {fact} [/INST] Prompt: {prompt} {tokenizer.eos_token}{tokenizer.bos_token}"
                    )
            if error_analysis:
                messages.append(
                    f"[INST] {fact_a} [/INST] Prompt: {prompt_b}"
                )
            elif realistic:
                # take the example with the highest similarity
                new_fact = icl_examples[0].split('Prompt: ')[0]#"New Fact: Not Available\n\n"
                # use it for the user question
                messages.append(
                    f"[INST] {new_fact} [/INST] Prompt: {q}"
                )
            else:
                messages.append(
                    f"[INST] New Fact: {sample['prompt']} {sample['target']}\n [/INST] Prompt: {q}"
                )
            inputs = tokenizer(''.join(messages), return_tensors='pt', padding=True)
        else:
            inputs = tokenizer(f"[INST] <<SYS>>\nAnswer with the name only.\n<</SYS>>\n\n{q} [/INST]", return_tensors='pt', padding=True)
    else:
        if icl_examples is not None and hparams.alg_name == "IKE":
            if error_analysis:
                q = ''.join(icl_examples + [f'{fact_a}Prompt: {prompt_b}'])
            elif realistic:
                # take the example with the highest similarity
                new_fact = icl_examples[0].split('Prompt: ')[0]#"New Fact: Not Available\n\n"
                # use it for the user question
                q = ''.join(icl_examples + [f'{new_fact}Prompt: {q}'])
            else:
                q = ''.join(icl_examples + [f'New Fact: {sample["prompt"]} {sample["target"]}\nPrompt: {q}'])
        inputs = tokenizer(q, return_tensors='pt', padding=True)

    if update_questions:
        res["questions"][qt] = tokenizer.batch_decode(inputs.input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    # Restore the original examples
    if error_analysis:
        icl_examples[-1 if realistic else -2] = original_icl_example

    return inputs


def generate_sample_answers(
        sample: dict, 
        answers: dict, 
        editor: BaseEditor,
        train_ds: ZsreDataset,
        tokenizer, 
        model, 
        device, 
        max_length: int,
        hparams,
        error_analysis: bool = False,
        realistic: bool = False
    ):
    domain, element, attribute = sample["domain"], sample["element"], sample["attribute"]

    if domain not in answers:
        answers[domain] = {}

    if element not in answers[domain]:
        answers[domain][element] = {}

    if attribute is not None and attribute not in answers[domain][element]:
        answers[domain][element][attribute] = {}

    res = {
        "questions": sample["questions"],
        "answers": {}
    }

    for qt, q in sample["questions"].items():
        if hparams.alg_name == "IKE":
            icl_examples = return_ike_examples(sample, q, editor, train_ds, realistic)
        else:
            icl_examples = None

        inputs = encode_inputs(
            q,
            qt,
            res,
            tokenizer,
            hparams,
            sample,
            update_questions=True,
            icl_examples=icl_examples,
            error_analysis=error_analysis,
            realistic=realistic
        ).to(device)

        if hparams.alg_name == "MEND":
            with torch.no_grad():
                outputs = model(**inputs)
                for _ in range(max_length):
                    outputs = model(**inputs)
                    if type(outputs) is torch.Tensor:
                        logits = outputs
                    else:
                        logits = outputs.logits
                    generate_ids = torch.argmax(logits, dim=-1).squeeze().detach().cpu().numpy().tolist()
                    if generate_ids[-1] == tokenizer.eos_token_id:
                        break
                    q = q + tokenizer.decode(generate_ids[-1])
                    inputs = encode_inputs(q, qt, res, tokenizer, hparams, sample, update_questions=False)
            res["answers"][qt] = tokenizer.batch_decode(
                inputs.input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
        else:
            generate_ids = model.generate(
                **inputs,
                max_new_tokens=max_length,
                pad_token_id=tokenizer.eos_token_id,
            )
            res["answers"][qt] = tokenizer.batch_decode(
                generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

    if attribute is not None:
        answers[domain][element][attribute] = res
    else:
        answers[domain][element] = res


def create_ike_train_dataset(dataset: dict) -> List[Dict[str, str]]:

    # first separate facts per domain so we can sample locality inputs from other domains
    facts_per_domain = {}
    for sample in dataset:
        domain = sample["domain"]
        if domain not in facts_per_domain:
            facts_per_domain[domain] = []
        # we use 'generic' prompts for the new facts
        facts_per_domain[domain].append(
            (sample["questions"]["generic"], sample["target"])
        )
    
    train_dataset = []
    for sample in dataset:
        # take locality facts from another domain
        locality_facts = [
            p for domain, prompts in facts_per_domain.items()
            for p in prompts
            if domain != sample["domain"]
        ]
        # randomly sample one locality fact
        locality_prompt, locality_gt = random.choice(locality_facts)

        train_data = {
            "prompt": sample["questions"]["generic"], # we use 'generic' prompts for the new facts
            "target_new": sample["target"],
            "rephrase_prompt": sample["prompt"], # we use editing prompts for paraphrases
            "locality_prompt": locality_prompt,
            "locality_ground_truth": locality_gt,
        }

        train_dataset.append(train_data)

    return train_dataset