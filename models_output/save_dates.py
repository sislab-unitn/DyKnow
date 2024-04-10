import argparse
import os
from argparse import Namespace


from utils import load_json, dump_json
from analyze_replies import analyze_replies


def save_dates(results_folder: str, question_path: str):
    """
    Save the dates extracted from the analysis results to a JSON file.

    Args:
        results_folder (str): The path to the folder containing the analysis results.
        question_path (str): The path to the question file.

    Returns:
        None
    """
    # make sure the _analysis.json files are there
    analyze_replies(results_folder, question_path)

    for file_name in os.listdir(results_folder):
        if not file_name.endswith("_analysis.json"):
            continue

        domain = file_name.split("_analysis.json")[0]
        analysis = load_json(os.path.join(results_folder, file_name))
        years = {}
        for question_type, answers_type in analysis.items():
            for answer_type, answers in answers_type.items():
                if answer_type in ["correct", "outdated"]:
                    for ans in answers:
                        assert (
                            len(ans["matched_answers"]) == 1
                        ), f"More predictions for {results_folder} about {ans['element']} --- {ans['attribute']}: {ans['matched_answers']}"
                        year = ans["matched_answers"].pop()[-1]
                        element = ans["element"]
                        attribute = ans["attribute"]
                        if element not in years:
                            years[element] = {}
                        if attribute is None:
                            years[element][question_type] = year
                        else:
                            if attribute not in years[element]:
                                years[element][attribute] = {}
                            years[element][attribute][question_type] = year
        dump_json(os.path.join(results_folder, f"{domain}_dates.json"), years)


def get_args() -> Namespace:
    """
    Parse command line arguments.

    Returns
    -------
    parsed_args: Namespace instance
        Parsed arguments passed through command line.
    """

    parser = argparse.ArgumentParser(
        prog="python -m save_dates",
        description="Save the dates based on the analysis of the generated answers.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # arguments of the parser
    parser.add_argument(
        "results_dir",
        metavar="DIR_NAME",
        type=str,
        help="Folder containing the generated answers from a model.",
    )
    parser.add_argument(
        "--question-path",
        metavar="FILE_PATH",
        type=str,
        default="../grc_generated.json",
        help="Path to the file containing Q&A.",
    )

    # parse arguments
    parsed_args = parser.parse_args()

    return parsed_args


if __name__ == "__main__":
    args = get_args()
    save_dates(args.results_dir, args.question_path)
