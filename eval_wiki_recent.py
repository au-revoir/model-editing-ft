import collections
import numpy as np
import torch
from itertools import chain
from scipy.stats import hmean
from tqdm import tqdm

def compute_wiki_recent_predictions(model, tokenizer, data):

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

        # Form a list of lists of prefixes to test.
        prob_prompts = [ 
            rewrite_prompts,
            paraphrase_prompts,
        ]   
        # Flatten all the evaluated prefixes into one list.
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

        which_correct = [
            [1 for _ in range(len(neighborhood_prompts))],
        ]
        neighborhood_probs, neighborhood_correct = test_batch_prediction( 
            model,
            tokenizer,
            [
                el["prompt"].format(record["requested_rewrite"])
                for el in neighborhood_prompts
            ],
            list(chain(*which_correct)),
            [el["target_new"] for el in neighborhood_prompts],
            [el["target_true"] for el in neighborhood_prompts]
        )

        probs = stuff_probs 

        # Unflatten the results again into a list of lists.
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
        ret["neighborhood_prompts_probs"] = neighborhood_probs

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

def test_batch_prediction(model, tok, prefixes, which_correct, target_new, target_true):
    prefix_lens = [len(n) for n in tok(prefixes)["input_ids"]]
    prompt_tok = tok(
        [
            f"{prefix} {suffix}"
            for prefix in prefixes
            for suffix in [target_new, target_true]
        ],
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    a_tok = [tok(f" {n_1}")["input_ids"] for n in target_new for n_1 in n]
    b_tok = [tok(f" {n_1}")["input_ids"] for n in target_true for n_1 in n]

    choice_a_len = [len(n) for n in a_tok]
    choice_b_len = [len(n) for n in b_tok]

    with torch.no_grad():
        logits = model(**prompt_tok).logits

    probs = np.zeros((logits.size(0),), dtype=np.float32)
    targets_correct = []


    for i in range(logits.size(0)):
        cur_len = choice_a_len[i] if i % 2 == 0 else choice_b_len[i]

        # Compute suffix probabilities
        for j in range(cur_len):
            cur_tok = (a_tok[i] if i % 2 == 0 else b_tok[i])[j]
            probs[i] += -torch.nn.functional.log_softmax(
                logits[i, prefix_lens[i // 2] + j - 1, :], dim=0
            )[cur_tok].item()
        probs[i] /= cur_len

        # Compute accuracy on new targets
        if (which_correct[i // 2] == 0 and i % 2 == 0) or (
            which_correct[i // 2] == 1 and i % 2 == 1
        ):
            correct = True
            for j in range(cur_len):
                cur_tok = (a_tok[i] if i % 2 == 0 else b_tok[i])[j]

                if logits[i, prefix_lens[i // 2] + j - 1, :].argmax().item() != cur_tok:
                    correct = False
                    break
            targets_correct.append(correct)

    return [
        {"target_new": probs[i].item(), "target_true": probs[i + 1].item()}
        for i in range(0, len(probs), 2)
    ], targets_correct

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
    for key in ["rewrite", "paraphrase"]:
        sum_key = f"{key}_acc"
        key = f"{key}_prompts_correct"

        if key not in results:
            continue

        cur_sum[sum_key].append(np.mean(results[key]))

    sum_key_discrete = f"neighborhood_success" 
    key = "neighborhood_prompts_probs"
    cur_sum[sum_key_discrete].append(
        np.mean(
            [
                x["target_true"] < x["target_new"]
                for x in results[key]
            ]
        )
    )
    return cur_sum["rewrite_acc"][0], cur_sum["paraphrase_acc"][0], cur_sum["neighborhood_success"][0]
