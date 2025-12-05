import os
from transformers import RobertaTokenizer, T5ForConditionalGeneration
import torch
from torch import cuda

from pathlib import Path
import re

DATA_PATH = "../../data/pddl"
SAVE_PATH = "../../data/plansformer"

abs_data_path = os.path.abspath(DATA_PATH)
abs_save_path = os.path.abspath(SAVE_PATH)


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

    flag = 1
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

    if domain == "blocks":
        for idx, obj in enumerate(objs, start=1):
            mapping[obj] = f"b{idx}"
        return mapping

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


def main():
    model_path = "/anvil/projects/x-nairr250014/plansformer/codet5-base/model_files"
    device = "cuda" if cuda.is_available() else "cpu"
    tokenizer = RobertaTokenizer.from_pretrained(model_path, local_files_only=True)
    model = T5ForConditionalGeneration.from_pretrained(
        model_path, local_files_only=True
    )
    model = model.to(device)
    domain = None
    for dirpath, dnames, fnames in os.walk(abs_data_path):
        if "domain.pddl" in fnames:
            domain = os.path.join(dirpath, "domain.pddl")

        p_files = []

        for fname in fnames:
            if re.fullmatch(
                r"^.+/test-interpolation/.+\.pddl$", os.path.join(dirpath, fname)
            ) or re.fullmatch(
                r"^.+/test-extrapolation/.+\.pddl$", os.path.join(dirpath, fname)
            ):
                p_files.append(
                    (
                        os.path.join(
                            dirpath,
                            fname,
                        )
                    )
                )

        for p_file in p_files:
            p_file = Path(p_file)
            text = p_file.read_text()
            domain_name = re.findall(r"(?<=domain )\w+", text)
            domain_name = domain_name[0]

            # Parse objects into the correct format for plansformer
            objs = parse_objects_from_pddl(text)
            mapping = build_mapping(objs, domain_name)
            converted = replace_identifiers(text, mapping)
            prompt = get_prompt(
                domain_file=domain,
                problem_file=converted,
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
            )
            predicted_plan = tokenizer.decode(
                generated_ids[0], skip_special_tokens=True
            )

            # Destination
            dest = os.path.join(abs_save_path, p_file.removeprefix(abs_data_path + "/"))

            os.makedirs(os.path.dirname(dest), exist_ok=True)

            with open(dest, "w", encoding="utf-8") as out_f:
                out_f.write(predicted_plan.strip() + "\n")
                # out_f.write("Hello_world" + "\n")


if __name__ == "__main__":
    main()
