from transformers import RobertaTokenizer, T5ForConditionalGeneration
import torch
from torch import cuda


def main():
    model_path = "model_files"
    device = "cuda" if cuda.is_available() else "cpu"
    tokenizer = RobertaTokenizer.from_pretrained(model_path, local_files_only=True)
    model = T5ForConditionalGeneration.from_pretrained(
        model_path, local_files_only=True
    )
    model = model.to(device)
    problem = """
    ontable b1, clear b1, ontable b2, clear b2, ontable b3, clear b3, ontable b4, clear b4
    handempty, ontable b1, on b2 b3, clear b2, ontable b3, on b4 b1, clear b4
     pick-up  clear x, ontable x, handempty  not ontable x, not clear x, not handempty, holding x
     put-down  holding x  not holding x, clear x, handempty, ontable x
     stack  holding x, clear y  not holding x, not clear y, clear x, handempty, on x y
     unstack  on x y, clear x, handempty  holding x, clear y, not clear x, not handempty, not on x y"
    """
    problem = "<GOAL> on b1 b2, on b2 b3, ontable b3, on b4 b1, clear b4 <INIT> handempty, ontable b1, clear b1, on b2 b3, ontable b3, on b4 b2, clear b4 <ACTION> pick-up <PRE> clear x, ontable x, handempty <EFFECT> not ontable x, not clear x, not handempty, holding x <ACTION> put-down <PRE> holding x <EFFECT> not holding x, clear x, handempty, ontable x <ACTION> stack <PRE> holding x, clear y <EFFECT> not holding x, not clear y, clear x, handempty, on x y <ACTION> unstack <PRE> on x y, clear x, handempty <EFFECT> holding x, clear y, not clear x, not handempty, not on x y"

    input_ids = tokenizer.encode(problem, return_tensors="pt").to(
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
    predicted_plan = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print(predicted_plan)


if __name__ == "__main__":
    main()
