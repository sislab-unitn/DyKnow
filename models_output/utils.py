import json
from typing import Any, Counter, Dict, List


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


ADDITIONAL_BITS = {
    "athletes_byPayment": [
        "[\w]?F(\.)?C(\.)?[\w]?",  # FC, F.C., AFC, ... for Football
        "[\w]?C(\.)?F(\.)?[\w]?",  # CF, ... for Football
        "[\w]?F(\.)?K(\.)?[\w]?",  # FK, ... for Football
        "[\w]?A(\.)?S(\.)?[\w]?",  # AS, ... for Football
        "[\w]?S(\.)?V(\.)?[\w]?",  # SV, ... for Football
        "[\w]?B(\.)?C(\.)?[\w]?",  # BC, ... for Basketball
        "[\w](\.)[\w](\.)",  # General regex for to remove two letter acronyms (with )
        "football",
        "(t|T)eam",
        "association",
        "men's",
        "basketball",
        "F1",
        "(S|s)cuderia",
        "(R|r)acing",
        "A$",
    ],
    "organizations": ["(C|c)ity of"],
}


def get_questions(
    grc_path: str, prompt: str = ""
) -> Dict[str, Dict[str, Dict[str, List[str]]]]:

    with open(grc_path, "r") as f:
        grc = json.load(f)

    questions: Dict[str, Dict[str, Dict[str, List[str]]]] = {}

    # Check catgories that have inner attributes
    for category, elements in grc.items():
        if category in ["athletes_byPayment", "companies_byRevenue"]:
            continue

        if category not in questions:
            questions[category] = {}

        for element, attributes in elements.items():
            for attribute, grc_elem in attributes.items():
                question_types = list(grc_elem["questions"].keys())
                key = f"{element}|{attribute}"
                assert key not in questions[category], f"{key} already used"
                questions[category][key] = {
                    "question_types": question_types,
                    "questions": [
                        " ".join([q, prompt]) if prompt else q
                        for q in grc_elem["questions"].values()
                    ],
                }

    # Check categories that only have elements
    for category, elements in grc.items():
        if category not in ["athletes_byPayment", "companies_byRevenue"]:
            continue

        if category not in questions:
            questions[category] = {}

        for element, grc_elem in elements.items():
            question_types = list(grc_elem["questions"].keys())
            key = f"{element}"
            assert key not in questions[category], f"{key} already used"
            questions[category][key] = {
                "question_types": question_types,
                "questions": [
                    " ".join([q, prompt]) if prompt else q
                    for q in grc_elem["questions"].values()
                ],
            }

    return questions


def load_json(path: str) -> Any:
    with open(path, "r") as f:
        json_file = json.load(f)
    return json_file


def dump_json(path: str, obj: Any, indent: int = 4):
    with open(path, "w") as f:
        json.dump(obj, f, indent=indent)


def write_roman(num: int):
    ROMAN = {
        1000: "M",
        900: "CM",
        500: "D",
        400: "CD",
        100: "C",
        90: "XC",
        50: "L",
        40: "XL",
        10: "X",
        9: "IX",
        5: "V",
        4: "IV",
        1: "I",
    }

    def roman_num(num: int):
        for r in ROMAN.keys():
            x, y = divmod(num, r)
            yield ROMAN[r] * x
            num -= r * x
            if num <= 0:
                break

    return "".join([a for a in roman_num(num)])


def get_correct_year(question_years: Counter[int]):
    # first get the year based on majority voting, otherwise the latest starting time
    question_years = list(
        sorted(
            [(y, c) for y, c in question_years.items()],
            key=lambda x: (x[1], x[0]),
            reverse=True,
        )
    )
    candidate_year = question_years.pop()
    return candidate_year[0]
