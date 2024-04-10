import argparse
import os
import json

from argparse import Namespace

from utils import prepare_targets


def get_args() -> Namespace:
    """
    Parse command line arguments.

    Returns
    -------
    parsed_args: Namespace instance
        Parsed arguments passed through command line.
    """

    parser = argparse.ArgumentParser(
        prog="python -m generate_editing_dataset",
        description="Generate the dataset for editing a specific model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # arguments of the parser
    parser.add_argument(
        "path_to_questions",
        metavar="PATH",
        type=str,
        help="Path to the set of questions that should be updated.",
    )
    parser.add_argument(
        "--out-dir",
        metavar="PATH",
        default="editing_datasets",
        type=str,
        help="Path of the output directory",
    )
    parser.add_argument(
        "--prompt_type_per_domain",
        nargs=4,
        metavar=("COUNTRIES", "ATHLETES", "COMPANIES", "ORGANIZATIONS"),
        choices={"contextualized", "rephrased_1", "rephrased_2", "rephrased_3"},
        default=["rephrased_2", "rephrased_1", "rephrased_2", "rephrased_3"],
        type=str,
        help="The type of prompt that should be asked for each domain.",
    )

    # parse arguments
    parsed_args = parser.parse_args()
    parsed_args.prompt_type_per_domain = {
        domain: pt for domain, pt in zip(
            ["countries_byGDP", "athletes_byPayment", "companies_byRevenue", "organizations"],
            parsed_args.prompt_type_per_domain
        )
    }

    return parsed_args
    

def main(args):
    with open(args.path_to_questions, "r") as f:
        questions = json.load(f)
    
    targets = {}
    for domain in questions:
        ans = prepare_targets(domain, questions)
        targets[domain] = ans

    dataset = []
    for domain, elements in questions.items():
        if domain not in ["countries_byGDP", "organizations"]:
            for element in elements:
                prompt = elements[element]["questions"][args.prompt_type_per_domain[domain]]
                subject = element
                if domain == "companies_byRevenue":
                    subject = f"CEO position at {subject}"
                assert subject in prompt, f"'{subject}' does not appear in '{prompt}'"
                target = targets[domain][element]
                dataset.append({
                    "prompt": prompt,
                    "target": target,
                    "subject": subject,
                    "domain": domain,
                    "element": element,
                    "attribute": None,
                    "prompt_type": args.prompt_type_per_domain[domain],
                    "questions": elements[element]["questions"],
                    "answers": elements[element]["answers"],
                })
        else:
            for element, attributes in elements.items():
                for attribute in attributes:
                    prompt = elements[element][attribute]["questions"][args.prompt_type_per_domain[domain]]
                    subject = attribute
                    if domain == "organizations":
                        if subject == "director / manager":
                            subject = "director"
                        subject = f"{subject} at {element}"
                    assert subject in prompt, f"'{subject}' does not appear in '{prompt}'"
                    target = targets[domain][element][attribute]
                    dataset.append({
                        "prompt": prompt,
                        "target": target,
                        "subject": subject,
                        "domain": domain,
                        "element": element,
                        "attribute": attribute,
                        "prompt_type": args.prompt_type_per_domain[domain],
                        "questions": elements[element][attribute]["questions"],
                        "answers": elements[element][attribute]["answers"],
                    })
    
    model_name = os.path.dirname(args.path_to_questions).split("/")[-1]
    base_path = os.path.join(args.out_dir, model_name)
    os.makedirs(base_path, exist_ok=True)
    dataset_path = os.path.join(base_path, "editing_dataset.json")
    with open(dataset_path, "w") as f:
        json.dump(dataset, f, indent=4)

if __name__ == "__main__":
    args = get_args()
    main(args)