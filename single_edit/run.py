import argparse
import os
import json
import random
import torch
import torch.nn as nn
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed, get_scheduler
from transformers import AdamW
from tqdm import tqdm
from data import LMDataset
from trainer import CustomCollatorWithPadding, Trainer
from calculate_rs_ge import perform_generation_test
from eval_counterfact import compute_counterfact_predictions
from attr_snippets import AttributeSnippets
from tfidf_stats import get_tfidf_vectorizer

no_deprecation_warning=True

def load_json(path):
    with open(path) as fp:
        data = json.load(fp)
    return data

def save_json(data, path):
    with open(path, "w") as fp:
        json.dump(data, fp)
    print(f"File saved to {path}")

def early_stopping(train_loss, threshold):
    counter = 0
    if train_loss < threshold:
        return True

def create_directory_with_next_integer(base_directory, additional_directory_name):
    new_directory = os.path.join(base_directory, additional_directory_name)
    os.makedirs(new_directory)
    return new_directory

def train(trainer, model, optimizer, collator, num_epochs, edit_example):
    model.train()
    for epoch in range(num_epochs)    :
        inputs_tokenized = collator.apply_tokenization(edit_example, epoch)

        loss = trainer.compute_loss(model, inputs_tokenized, ignore_template_labels=True)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
    return model 

def main(args):
    set_seed(42)
    base_directory = args.output_save_path
    save_path = create_directory_with_next_integer(base_directory, "gpu_" + str(args.gpu_id))

    model = AutoModelForCausalLM.from_pretrained(args.model_name, cache_dir="/data/local/gg676/pretrained_models", torch_dtype=torch.bfloat16, device_map=f"cuda:{args.gpu_id}")
    state_dict = model.state_dict()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    if args.generated_prepended_words_path is not None:
        prepended_words = load_json(args.generated_prepended_words_path)
    else:
        prepended_words = None
    
    dataset = LMDataset(args, model, tokenizer, prepended_words)
    data_transformed = dataset.get_dataset()


    snips = AttributeSnippets("data/attribute_snippets/attribute_snippets.json")
    vec = get_tfidf_vectorizer("data/tfidf")

    collator = CustomCollatorWithPadding(tokenizer)
    print("lr: ", float(args.lr))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr))

    num_training_steps = args.num_epochs * len(data_transformed)

    trainer = Trainer()

    scores_list = []
    ge_rs_list = []
    for edit_example, record in zip(data_transformed, dataset.data):
        case_id = record["case_id"]

        model = train(trainer, model, optimizer, collator, args.num_epochs, edit_example)
        if args.dataset_name == "counterfact": 
            model.eval()
            res_ge_rs = perform_generation_test(model, tokenizer, record, snips, vec)
            res_scores = compute_counterfact_predictions(model, tokenizer, record)

            item = {"case_id": case_id, "scores": res_scores, "fluency": res_ge_rs["ngram_entropy"], "consistency": res_ge_rs["reference_score"]}
        else:
            
            res_scores = compute_counterfact_predictions(model, tokenizer, record)

            item = {"case_id": case_id, "scores": res_scores}
        save_json(item, f"{save_path}/case_id_{case_id}.json")
        model.load_state_dict(state_dict)
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--data_start", type=int, required=True)
    parser.add_argument("--data_end", type=int, required=True)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--prompt_rewrite", action="store_true")
    parser.add_argument("--prompt_paraphrase_type", type=str, required=True)
    parser.add_argument("--prompt_neighborhood", action="store_true")
    parser.add_argument("--prompt_neighborhood_type", type=str, required=True)
    parser.add_argument("--generated_prepended_words_path", type=str, default=None)
    parser.add_argument("--num_epochs", type=int, required=True)
    parser.add_argument("--lr", type=str, required=True)
    parser.add_argument("--output_save_path", type=str, required=True)
    args = parser.parse_args()
    main(args)
