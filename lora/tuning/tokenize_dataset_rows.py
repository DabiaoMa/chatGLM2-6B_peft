import argparse
import json
from tqdm import tqdm

import datasets
import transformers


def preprocess(tokenizer, config, example, max_seq_length_prompt, max_seq_length_answer):
    prompt = example["content"]
    answer = example["summary"]

    a_ids = tokenizer.encode(text=prompt, add_special_tokens=False)
    b_ids = tokenizer.encode(text=answer, add_special_tokens=False)

    if len(a_ids) > max_seq_length_prompt - 1:
        a_ids = a_ids[: max_seq_length_prompt - 1]

    if len(b_ids) > max_seq_length_answer - 2:
        b_ids = b_ids[: max_seq_length_answer - 2]

    input_ids = tokenizer.build_inputs_with_special_tokens(a_ids, b_ids)

    context_length = input_ids.index(tokenizer.get_command("sop"))

    mask_position = context_length - 1
    labels = [-100] * context_length + input_ids[mask_position+1:]
    return {"input_ids": input_ids, "labels": labels}


def read_jsonl(path, max_seq_length_prompt, max_seq_length_answer):
    model_name = "../../model/" # chatglm2-6B model path
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True)
    config = transformers.AutoConfig.from_pretrained(
        model_name, trust_remote_code=True, device_map='auto')
    with open(path, "r") as f:
        for line in tqdm(f.readlines()):
            example = json.loads(line)
            feature = preprocess(tokenizer, config, example, max_seq_length_prompt, max_seq_length_answer)
            yield feature


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl_path", type=str, default="data/train.json")
    parser.add_argument("--save_path", type=str, default="data/train")
    parser.add_argument("--max_seq_length_prompt", type=int, default=4800) 
    parser.add_argument("--max_seq_length_anwser", type=int, default=1400) 
    args = parser.parse_args()

    dataset = datasets.Dataset.from_generator(
        lambda: read_jsonl(args.jsonl_path, args.max_seq_length_prompt, args.max_seq_length_anwser)
    )
    dataset.save_to_disk(args.save_path)


if __name__ == "__main__":
    main()
