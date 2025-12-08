import json
import os
from transformers import RobertaTokenizer, T5ForConditionalGeneration
import torch
from torch import cuda
import argparse
from typing import (
    List,
)

from pathlib import Path
import re
from code.common.utils import validate_plan


def find_parens(s):
    toret = {}
    pstack = []
    flag = 0
    for i, c in enumerate(s):
        if flag == 1 and len(pstack) == 0:
            return toret

        if c == "(":
            pstack.append(i)
            flag = 1
        elif c == ")":
            toret[pstack.pop()] = i
    return toret


def prompt_action(data):
    # print(data)
    string = "<ACTION> " + data.split("\n")[0].split(" ")[1].lower() + " "

    if ":precondition" in data:
        string += "<PRE> "
        pre_idx = data.find(":precondition")
        effect_idx = data.find(":effect")
        pre_data = data[pre_idx:effect_idx]
        pre_parens = find_parens(pre_data)
        # print(pre_data)
        for keys in sorted(pre_parens.keys()):
            temp_str = pre_data[keys : pre_parens[keys] + 1]
            if "(and" not in temp_str:
                string += temp_str.strip("(").strip(")").replace("?", "") + ", "

        string = string[:-2] + " "
        # print(string)

    if ":effect" in data:
        string += "<EFFECT> "
        effect_idx = data.find(":effect")
        effect_data = data[effect_idx:]
        effect_parens = find_parens(effect_data)

        check_st = 0
        check_end = 0
        flag = 0
        flag_forall = 0
        for keys in sorted(effect_parens.keys()):
            temp_str = effect_data[keys : effect_parens[keys] + 1]

            if "(forall" in temp_str[:8] and "(and" not in temp_str[:5]:
                for ind in range(len(temp_str)):
                    if ind < len(temp_str) - 2:
                        if temp_str[ind] == ")" and temp_str[ind + 2] == "(":
                            temp_str = temp_str[:ind] + ", " + temp_str[ind + 2 :]

                string += (
                    "("
                    + temp_str.replace("?", "")
                    .replace("(", "")
                    .replace(")", "")
                    .replace("and", "")
                    .replace("\n", ",")
                    .strip()
                    + ")"
                    + ", "
                )
                flag_forall = 1

            elif flag_forall == 0:
                if flag == 1:
                    if keys > check_st and keys < check_end:
                        continue

                if "(and" not in temp_str:
                    string += (
                        temp_str.replace("(", "").replace(")", "").replace("?", "")
                        + ", "
                    )

                    if "(not" in temp_str:
                        flag = 1
                        check_st = keys
                        check_end = effect_parens[keys]

        string = string[:-2] + " "
    return string


def prompt_problem(data):
    if "(:init" in data:
        string = "<INIT> "

        parens = find_parens(data)
        check_st = 0
        check_end = 0
        flag = 0

        for keys in sorted(parens.keys()):
            temp_str = data[keys : parens[keys] + 1]

            if flag == 1:
                if keys > check_st and keys < check_end:
                    continue

            if "(:init" not in temp_str:
                string += temp_str.replace("(", "").replace(")", "") + ", "

                if "(not" in temp_str:
                    flag = 1
                    check_st = keys
                    check_end = parens[keys]

        return string[:-2] + " "

    elif "(:goal" in data:
        string = "<GOAL> "

        parens = find_parens(data)
        check_st = 0
        check_end = 0
        flag = 0

        for keys in sorted(parens.keys()):
            temp_str = data[keys : parens[keys] + 1]

            if flag == 1:
                if keys > check_st and keys < check_end:
                    continue

            if ("(:goal" and "(and") not in temp_str:
                string += temp_str.replace("(", "").replace(")", "") + ", "

                if "(not" in temp_str:
                    flag = 1
                    check_st = keys
                    check_end = parens[keys]
        return string[:-2] + " "


def get_prompt(domain_file, problem_data):
    with open(domain_file, "r") as f:
        domain_data = f.read()

    domain_name = re.findall(r"(?<=domain )\w+", domain_data)
    domain_name = domain_name[0]

    # flag = 1
    # domain_name += '_rao'
    # print("Domain: " + domain_name)

    # continue

    idx = [m.start() for m in re.finditer(pattern=":action", string=domain_data)]

    domain_string = ""

    for id in range(len(idx)):
        # print(domain_data[idx[id]-1: idx[id+1]-1])
        if id != len(idx) - 1:
            domain_string += (
                prompt_action(domain_data[idx[id] - 1 : idx[id + 1] - 1]).strip() + " "
            )
        else:
            domain_string += prompt_action(domain_data[idx[id] - 1 :])

    problem_data = problem_data.replace("(:INIT", "(:init").replace("(:GOAL", "(:goal")

    init_ind = problem_data.find("(:init")
    goal_ind = problem_data.find("(:goal")

    problem_string = ""
    problem_string += prompt_problem(problem_data[goal_ind:])
    problem_string += prompt_problem(problem_data[init_ind:goal_ind])

    return problem_string + domain_string


def parse_objects_from_pddl(text):
    # Find the :objects (...) section; allow objects on multiple lines
    # Regex is AI generated, validated with online tools
    m = re.search(r"\(:objects\s+([^\)]*?)\)", text, flags=re.DOTALL)
    if not m:
        return []
    objs_raw = m.group(1).strip()
    # split on whitespace
    objs = re.split(r"\s+", objs_raw)
    # filter empty strings
    objs = [o for o in objs if o]
    return objs


def build_mapping(objs, domain: str):
    mapping = {}

    # Plansformer was trained on "b1", "b2", ...
    if domain == "blocks":
        for idx, obj in enumerate(objs, start=1):
            mapping[obj] = f"b{idx}"
        return mapping

    # Doing the same for gripper, although plansformer was trained on gripper with two robots
    elif domain == "gripper":
        room_objs = [o for o in objs if o.startswith("room")]

        for idx, obj in enumerate(room_objs, start=1):
            mapping[obj] = f"room{idx}"

        # Identity mapping for all others
        for o in objs:
            mapping.setdefault(o, o)

        return mapping
    return None


def replace_identifiers(text, mapping):
    # Replace whole-word occurrences using regex word boundaries.
    # Sort keys by length descending to avoid partial overlaps (rare but safe).
    if not mapping:
        return text

    keys = sorted(mapping.keys(), key=len, reverse=True)

    def repl(match):
        token = match.group(0)
        return mapping.get(token, token)

    # Build a regex that matches any of the identifiers as whole words.
    pattern = r"\b(" + "|".join(re.escape(k) for k in keys) + r")\b"
    return re.sub(pattern, repl, text)


# Converts comma seperated plan to a list of actions with parentheses around actions
def plan_to_list(plan: str) -> List[str]:
    return [f"({a.strip()})" for a in plan.split(",")]


def inference(
    val_path: str,
    data_path: str,
    save_path: str,
    model_path: str,
):
    device = "cuda" if cuda.is_available() else "cpu"
    tokenizer = RobertaTokenizer.from_pretrained(model_path, local_files_only=True)
    model = T5ForConditionalGeneration.from_pretrained(
        model_path, local_files_only=True
    ).to(device)

    model.eval()
    # Don't @ me about this name.
    domain_file_local = None
    for dirpath, dnames, fnames in os.walk(data_path):
        # I don't like this
        if "domain.pddl" in fnames:
            domain_file_local = os.path.join(dirpath, "domain.pddl")

        # This will only have data in it if we are in the test-interpolation dir
        problem_file_paths = []
        problem_file_names = []

        problem_type = None
        domain_name = None

        for fname in fnames:
            abs_fname_path = os.path.join(dirpath, fname)

            m = re.fullmatch(
                r".*[/\\]pddl[/\\]([^/\\]+)[/\\](test-(?:interpolation|extrapolation)|validation)[/\\].*\.pddl$",
                abs_fname_path,
            )
            if m:
                domain_name = m.group(1)
                problem_type = m.group(2)

                problem_file_paths.append(abs_fname_path)
                problem_file_names.append(fname)

        # If we didn't find any problem files, we're in the wrong dir
        if len(problem_file_paths) < 1:
            continue

        dest = os.path.join(save_path, f"{domain_name}_{problem_type}.json")
        os.makedirs(os.path.dirname(dest), exist_ok=True)

        results = []

        for problem_file_path, problem_file_name in zip(
            problem_file_paths, problem_file_names
        ):
            assert problem_type and domain_name, (
                "Internal logic error, found problem files, but failed to parse the problem type or domain name"
            )
            with open(problem_file_path, "r", encoding="utf-8") as f:
                text = f.read()

            # Parse objects into the correct format for plansformer
            objs = parse_objects_from_pddl(text)
            mapping = build_mapping(objs, domain_name)
            converted_problem = replace_identifiers(text, mapping)
            prompt = get_prompt(
                domain_file=domain_file_local,
                problem_data=converted_problem,
            )

            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(
                device, dtype=torch.long
            )

            generated_ids = model.generate(
                input_ids,
                num_beams=2,
                max_length=input_ids.shape[-1] + 2,
                repetition_penalty=2.5,
                length_penalty=1.0,
                early_stopping=False,
                do_sample=True,
            )
            predicted_plan = tokenizer.decode(
                generated_ids[0], skip_special_tokens=True
            )

            # Inverse the mapping and apply it to the generated plan
            converted_plan = None
            if mapping:
                inv_mapping = {v: k for k, v in mapping.items()}
                converted_plan = plan_to_list(
                    replace_identifiers(predicted_plan, inv_mapping)
                )
            else:
                converted_plan = plan_to_list(predicted_plan)

            result = {
                "problem": problem_file_name,
                "plan": converted_plan,
                "plan_len": len(converted_plan),
            }

            # Plan validation
            is_solved, is_executable = validate_plan(
                domain_path=domain_file_local,
                problem_path=problem_file_path,
                plan_actions=converted_plan,
                val_path=val_path,
            )

            result["val_solved"] = is_solved
            result["val_executable"] = is_executable
            result["solved"] = is_solved

            results.append(result)

        # Print stats about results
        print("=" * 40)
        print(f"{domain_name}_{problem_type} results")
        print(f"Number of results: {len(results)}")
        print(f"Number of solved: {len([r for r in results if r['solved']])}")
        print(
            f"Number of executable: {len([r for r in results if r['val_executable']])}"
        )

        with open(dest, "w", encoding="utf-8") as out_f:
            json.dump(results, out_f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--val_path",
        default=os.environ.get("VAL_PATH", None),
        help="Path to VAL binary",
    )
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--model_path", required=True)

    args = parser.parse_args()

    assert args.val_path, "Validation binary path not provided"

    # Convert to absolute
    args.val_path = str(Path(args.val_path).expanduser().resolve())
    args.data_path = str(Path(args.data_path).expanduser().resolve())
    args.save_path = str(Path(args.save_path).expanduser().resolve())
    args.model_path = str(Path(args.model_path).expanduser().resolve())

    inference(**vars(args))
