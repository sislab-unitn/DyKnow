import argparse
import json
import os
from argparse import Namespace

from EasyEdit.easyeditor import EditTrainer, MENDTrainingHparams, SERACTrainingHparams, ZsreDataset


def get_args() -> Namespace:
    """
    Parse command line arguments.

    Returns
    -------
    parsed_args: Namespace instance
        Parsed arguments passed through command line.
    """

    parser = argparse.ArgumentParser(
        prog="python -m train_editing_method",
        description="Train an editing method.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # arguments of the parser
    parser.add_argument(
        "edit_alg",
        metavar="EDIT_ALG_NAME",
        type=str,
        choices={"MEND", "SERAC"},
        help="Editing algorithm to update a model.",
    )
    parser.add_argument(
        "hparams_path",
        metavar="HPARAMS_PATH",
        type=str,
        help="Path to the hparams path with the training configuration.",
    )
    parser.add_argument(
        "--zsre_data_path",
        metavar="ZSRE_DATA_PATH",
        default="data/zsre",
        type=str,
        help="Path to zsre train and eval dataset.",
    )
    parser.add_argument(
        "--out-dir",
        metavar="DIR_NAME",
        type=str,
        default="results",
        help="Destination folder to save the generation results.",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="If set, parallelize on available gpus.",
    )

    # parse arguments
    parsed_args = parser.parse_args()

    if parsed_args.edit_alg == "MEND":
        parsed_args.edit_alg = MENDTrainingHparams
    elif parsed_args.edit_alg == "SERAC":
        parsed_args.edit_alg = SERACTrainingHparams
    else:
        raise NotImplementedError

    return parsed_args

def main(args: Namespace):

    # Loading config
    hparams = args.edit_alg.from_hparams(args.hparams_path)
    hparams.model_parallel = args.parallel

    # Training and Eval set
    train_ds = ZsreDataset(
        os.path.join(args.zsre_data_path, 'zsre_mend_train.json'), config=hparams
    )
    eval_ds = ZsreDataset(
        os.path.join(args.zsre_data_path,'zsre_mend_eval.json'), config=hparams
    )
    
    # Trainer
    trainer = EditTrainer(
        config=hparams,
        train_set=train_ds,
        val_set=eval_ds
    )

    trainer.run()

if __name__ == "__main__":
    args = get_args()
    main(args)