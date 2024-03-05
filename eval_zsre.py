import collections
import numpy as np
import torch
from itertools import chain
from scipy.stats import hmean
from tqdm import tqdm

def compute_zsre_predictions(model, tokenizer, data):
    rewrite_success_count = 0
    paraphrase_success_count = 0
    neighborhood_success_count = 0

    rewrite_success_count_list = []
    paraphrase_success_count_list = []
    neighborhood_success_count_list = []

    for record in tqdm(data):

        subject, target_new, target_true = (
            record["requested_rewrite"][x] for x in ["subject", "target_new", "target_true"]
        )
        rewrite_prompts = [record["requested_rewrite"]["prompt"].format(subject)]
        paraphrase_prompts = record["paraphrase_prompts"]
        neighborhood_prompts = record["neighborhood_prompts"]


        prob_prompts = [
        rewrite_prompts,
        paraphrase_prompts,
        ]
        target_tok = tokenizer(" " + target_new["str"])["input_ids"]
        inp_prompts_og = list(chain(*prob_prompts))

        inp_prompts = [
            el + tokenizer.decode(target_tok[:i])
            for el in inp_prompts_og
            for i in range(len(target_tok))
        ]
        inp_targets = [
            tokenizer.decode(target_tok[i])
            for _ in range(len(inp_prompts_og))
            for i in range(len(target_tok))
        ]

        stuff_probs = test_batch_prediction_acc(model, tokenizer, inp_prompts, inp_targets)

        neighborhood_correct = test_batch_prediction_acc(
            model,
            tokenizer,
            [
                el["prompt"].format(record["requested_rewrite"])
                for el in neighborhood_prompts
            ],
            [el["target"] for el in neighborhood_prompts],
        )

        probs = stuff_probs + neighborhood_correct

        cutoffs = [0] + np.cumsum(
            [l * len(target_tok) for l in map(len, prob_prompts)]
        ).tolist()
        ret_probs = [probs[cutoffs[i - 1] : cutoffs[i]] for i in range(1, len(cutoffs))]

        ret = {
            f"{key}_correct": ret_probs[i]
            for i, key in enumerate(
                [
                    "rewrite_prompts",
                    "paraphrase_prompts",
                ]
            )
        }

        ret["neighborhood_prompts_correct"] = neighborhood_correct
        
        rewrite_success, paraphrase_success, neighborhood_success  = calculate_accuracy(ret)
    
        rewrite_success_count += rewrite_success
        paraphrase_success_count += paraphrase_success
        neighborhood_success_count += neighborhood_success


        rewrite_success_count_list.append(rewrite_success)
        paraphrase_success_count_list.append(paraphrase_success)
        neighborhood_success_count_list.append(neighborhood_success)

    rewrite_mean_std = (np.mean(rewrite_success_count_list), np.std(rewrite_success_count_list))
    paraphrase_mean_std = (np.mean(paraphrase_success_count_list), np.std(paraphrase_success_count_list))
    neighborhood_mean_std = (np.mean(neighborhood_success_count_list), np.std(neighborhood_success_count_list))



    print("Rewrite Accuracy: ", tuple(np.around(z * 100, 2) for z in rewrite_mean_std))
    print("Paraphrase Accuracy: ", tuple(np.around(z * 100, 2) for z in paraphrase_mean_std))
    print("Neighborhood Accuracy: ", tuple(np.around(z * 100, 2) for z in neighborhood_mean_std))

    print("Rewrite Accuracy: ", rewrite_success_count / len(data))
    print("Paraphrase Accuracy: ", paraphrase_success_count / len(data))
    print("Neighborhood Accuracy: ", neighborhood_success_count / len(data))
    print("Harmonic Score: ", hmean([rewrite_success_count / len(data), paraphrase_success_count / len(data), neighborhood_success_count / len(data)]))

def test_batch_prediction_acc(model, tok, prompts, target):
    prompt_tok = tok(
        prompts,
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    with torch.no_grad():
        logits = model(**prompt_tok).logits
        last_non_masked = prompt_tok["attention_mask"].sum(1) - 1
        to_gather = last_non_masked.unsqueeze(1).repeat(1, logits.size(-1)).unsqueeze(1)
        gathered = torch.gather(logits, 1, to_gather).squeeze(1)
        ans = torch.argmax(gathered, dim=1)

        correct_id = tok(target, padding=True, return_tensors="pt").to("cuda")[
            "input_ids"
        ]
        correct_id = correct_id[:, 0].squeeze()

        return (ans == correct_id).detach().cpu().numpy().tolist()

def calculate_accuracy(results):
    cur_sum = collections.defaultdict(lambda: [])
    for key in ["rewrite", "paraphrase", "neighborhood"]:
        sum_key = f"{key}_acc"
        key = f"{key}_prompts_correct"

        if key not in results:
            continue

        cur_sum[sum_key].append(np.mean(results[key]))
    return cur_sum["rewrite_acc"][0], cur_sum["paraphrase_acc"][0], cur_sum["neighborhood_acc"][0]
