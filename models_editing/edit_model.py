import argparse
import json
import os
from argparse import Namespace

from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from EasyEdit.easyeditor import (
    BaseEditor, ROMEHyperParams, MEMITHyperParams, MENDHyperParams, 
    SERACHparams, IKEHyperParams, ZsreDataset
)
from EasyEdit.easyeditor.models.ike import encode_ike_facts
from utils import generate_sample_answers, create_ike_train_dataset


def get_args() -> Namespace:
    """
    Parse command line arguments.

    Returns
    -------
    parsed_args: Namespace instance
        Parsed arguments passed through command line.
    """

    parser = argparse.ArgumentParser(
        prog="python -m edit_model",
        description="Edit a model using a specific method.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # arguments of the parser
    parser.add_argument(
        "edit_alg",
        metavar="EDIT_ALG_NAME",
        type=str,
        choices={"ROME", "MEMIT", "MEND", "SERAC", "IKE"},
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
    parser.add_argument(
        "--data-dir",
        metavar="DIR_NAME",
        type=str,
        default="data",
        help="Data directory (used for IKE).",
    )
    parser.add_argument(
        "--error-analysis",
        action="store_true",
        help="If set, perform error analysis on IKE.",
    )
    parser.add_argument(
        "--realistic",
        action="store_true",
        help="If set, perform use IKE in a realistic way.",
    )

    # parse arguments
    parsed_args = parser.parse_args()

    if parsed_args.edit_alg == "ROME":
        parsed_args.edit_alg = ROMEHyperParams
    elif parsed_args.edit_alg == "MEMIT":
        parsed_args.edit_alg = MEMITHyperParams
    elif parsed_args.edit_alg == "MEND":
        parsed_args.edit_alg = MENDHyperParams
    elif parsed_args.edit_alg == "SERAC":
        parsed_args.edit_alg = SERACHparams
    elif parsed_args.edit_alg == "IKE":
        parsed_args.edit_alg = IKEHyperParams
    else:
        raise NotImplementedError

    return parsed_args

def main(args: Namespace):

    # Loading config
    hparams = args.edit_alg.from_hparams(args.hparams_path)
    hparams.model_parallel = args.parallel
    
    # Editor
    editor = BaseEditor.from_hparams(hparams)
    # Tokenizer
    tokenizer = editor.tok

    # Dataset
    with open(args.edit_dataset_path, "r") as f:
        dataset = json.load(f)
    
    alg_folder = hparams.alg_name
    if hparams.alg_name == "IKE":
        if args.realistic:
            alg_folder = f"{alg_folder}_REALISTIC"
        if args.error_analysis:
            alg_folder = f"{alg_folder}_ERR_ANALYSIS"

    args.out_dir = os.path.join(
        args.out_dir,
        alg_folder, 
        "batch_editing" if args.batch_editing else "single_edits",
        hparams.model_name.split('/')[-1]
    )
    os.makedirs(args.out_dir, exist_ok=True)

    if hparams.alg_name == 'IKE':
        #train_data_path = os.path.join(args.data_dir, 'zsre/zsre_mend_train_10000.json')
        #train_ds = ZsreDataset(train_data_path)
        train_ds = create_ike_train_dataset(dataset)
        sentence_model = SentenceTransformer(hparams.sentence_model_name).to(f'cuda:{hparams.device}')
        #encode_ike_facts(sentence_model, train_ds, hparams)
        encode_ike_facts(sentence_model, train_ds, hparams)
    else:
        train_ds = None

    answers = {}
    ## edit descriptor: prompt that you want to edit
    prompts = []
    ## subject: suject in the prompt (for ROME and MEMIT)
    subjects = []
    ## edit target: expected output
    targets = []

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

    if hparams.alg_name != 'IKE':
        if args.batch_editing:
            hparams.batch_size = len(subjects)
            metrics, edited_model, _ = editor.batch_edit(
                prompts=prompts,
                subject=subjects,
                target_new=targets,
                keep_original_weight=False,
                test_generation=args.test_generation,
                train_ds=train_ds
            )
        else:
            metrics, edited_model, _ = editor.edit(
                prompts=prompts,
                subject=subjects,
                target_new=targets,
                keep_original_weight=False,
                test_generation=args.test_generation,
                train_ds=train_ds
            )
        
        weight_dir = os.path.join(
            "edited_models",
            alg_folder, 
            "batch_editing" if args.batch_editing else "single_edits",
            hparams.model_name.split('/')[-1]
        )

        if args.save_model:
            edited_model.save_pretrained(weight_dir)
            editor.tok.save_pretrained(weight_dir)

        json.dump(metrics, open(os.path.join(args.out_dir, f'results.json'), 'w'), indent=4)

    for sample in tqdm(dataset, desc="Generating"):
        generate_sample_answers(
            sample,
            answers,
            editor,
            train_ds,
            tokenizer,
            edited_model if hparams.alg_name != 'IKE' else editor.model,
            f"cuda:{hparams.device}",
            args.max_length,
            hparams,
            args.error_analysis,
            args.realistic
        )

    for domain in answers:
        with open(os.path.join(args.out_dir, f"{domain}_answers.json"), "w") as f:
            json.dump(answers[domain], f, indent=4)

if __name__ == "__main__":
    args = get_args()
    main(args)