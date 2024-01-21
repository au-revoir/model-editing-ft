import collections
import numpy as np
import torch
from itertools import chain
from scipy.stats import hmean
from tqdm import tqdm

def compute_counterfact_predictions(model, tokenizer, data):
    #data = data[0]
    rewrite_success_count = 0
    paraphrase_success_count = 0
    neighborhood_success_count = 0

    rewrite_total = 0
    paraphrase_total = 0
    neighborhood_total = 0

    rewrite_diff_total = 0
    paraphrase_diff_total = 0
    neighborhood_diff_total = 0

    for record in tqdm(data):
        #record = data[0]
        subject, target_new, target_true = (
            record["requested_rewrite"][x] for x in ["subject", "target_new", "target_true"]
        )

        #print("target_new: ", target_new)
        #print("target_true: ", target_true)
        #quit()
        rewrite_prompts = [record["requested_rewrite"]["prompt"].format(subject)]
        paraphrase_prompts = record["paraphrase_prompts"]
        neighborhood_prompts = [i["prompt"] for i in record["neighborhood_prompts"]]
        #print("rewrite_prompts: ", rewrite_prompts)
        #print("paraphrase_prompts: ", paraphrase_prompts)
        #print("neighborhood_prompts: ", neighborhood_prompts)
        #quit()

        rewrite_total += len(rewrite_prompts)
        paraphrase_total += len(paraphrase_prompts)
        neighborhood_total += len(neighborhood_prompts)

        prob_prompts = [
            rewrite_prompts,
            paraphrase_prompts,
            neighborhood_prompts,
        ]

        probs = test_batch_prediction(
            model, tokenizer, list(chain(*prob_prompts)), target_new["str"], target_true["str"]
        )

        #print(probs)

        # Unflatten the results again into a list of lists.
        cutoffs = [0] + np.cumsum(list(map(len, prob_prompts))).tolist()
        ret_probs = [probs[cutoffs[i - 1] : cutoffs[i]] for i in range(1, len(cutoffs))]
        # Structure the restuls as a dictionary.
        ret = {
            f"{key}_probs": ret_probs[i]
            for i, key in enumerate(
                [
                    "rewrite_prompts",
                    "paraphrase_prompts",
                    "neighborhood_prompts",
                ]
            )
        }

        #print(ret)

        rewrite_success, paraphrase_success, neighborhood_success, rewrite_diff, paraphrase_diff, neighborhood_diff = calculate_accuracy(ret)

        rewrite_success_count += rewrite_success
        paraphrase_success_count += paraphrase_success
        neighborhood_success_count += neighborhood_success

        rewrite_diff_total += rewrite_diff
        paraphrase_diff_total += paraphrase_diff
        neighborhood_diff_total += neighborhood_diff



    print("\nRewrite Total: ", rewrite_total) 
    print("Paraphrase Total: ", paraphrase_total)
    print("Neighborhood Total: ", neighborhood_total)

    print("Rewrite Success: ", rewrite_success_count)
    print("Paraphrase Success: ", paraphrase_success_count) 
    print("Neighborhood Success: ", neighborhood_success_count)

    print("Rewrite Accuracy: ", rewrite_success_count / len(data)) 
    print("Paraphrase Accuracy: ", paraphrase_success_count / len(data))
    print("Neighborhood Accuracy: ", neighborhood_success_count / len(data))



    print("Rewrite Magnitude: ", rewrite_diff_total / len(data)) 
    print("Paraphrase Magnitude: ", paraphrase_diff_total / len(data))
    print("Neighborhood Magnitude: ", neighborhood_diff_total / len(data))

    print("Harmonic Score: ", hmean([rewrite_success_count / len(data), paraphrase_success_count / len(data), neighborhood_success_count / len(data)]))




def test_batch_prediction(
    model,
    tok,
    prefixes,
    target_new,
    target_true,
):
    """ """
    #print("prefixes: ", prefixes)
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

    a_tok, b_tok = (tok(f" {n}")["input_ids"] for n in [target_new, target_true])
    choice_a_len, choice_b_len = (len(n) for n in [a_tok, b_tok])

    with torch.no_grad():
        logits = model(**prompt_tok).logits

    results = np.zeros((logits.size(0),), dtype=np.float32)

    for i in range(logits.size(0)):
        cur_len = choice_a_len if i % 2 == 0 else choice_b_len
        for j in range(cur_len):
            cur_tok = (a_tok if i % 2 == 0 else b_tok)[j]
            results[i] += -torch.nn.functional.log_softmax(
                logits[i, prefix_lens[i // 2] + j - 1, :], dim=0
            )[cur_tok].item()
        results[i] /= cur_len

    return [
        {"target_new": results[i].item(), "target_true": results[i + 1].item()}
        for i in range(0, len(results), 2)
    ]


def calculate_accuracy(results):
    cur_sum = collections.defaultdict(lambda: [])
    for key in ["rewrite_prompts_probs", "paraphrase_prompts_probs"]:
        sum_key_discrete = f"{key.split('_')[0]}_success"
        sum_key_cont = f"{key.split('_')[0]}_diff"

        cur_sum[sum_key_discrete].append(
            np.mean(
                [
                    x["target_true"] > x["target_new"]
                    for x in results[key]
                ]
            )
        )
        cur_sum[sum_key_cont].append(
            np.mean(
                [
                    np.exp(-x["target_new"]) - np.exp(-x["target_true"])
                    for x in results[key]
                ]
            )
                    )
        
    
    # Probability metrics for which true should be lower (better) than new
    sum_key_discrete = f"neighborhood_success"
    sum_key_cont = f"neighborhood_diff"
    key = "neighborhood_prompts_probs"
    cur_sum[sum_key_discrete].append(
        np.mean(
            [
                x["target_true"] < x["target_new"]
                for x in results[key]
            ]
        )
    )
    cur_sum[sum_key_cont].append(
        np.mean(
            [
                np.exp(-x["target_true"]) - np.exp(-x["target_new"])
                for x in results[key]
            ]
        )
    )
    #print("cur_sum: ", cur_sum)
    #quit()
    return cur_sum["rewrite_success"][0], cur_sum["paraphrase_success"][0], cur_sum["neighborhood_success"][0], cur_sum["rewrite_diff"][0], cur_sum["paraphrase_diff"][0], cur_sum["neighborhood_diff"][0]
    #print("\ncur_sum: ", cur_sum)
    #print("\nRewrite Accuracy: ", cur_sum["rewrite_success"][0])
    #print("\nParaphrase Accuracy: ", cur_sum["paraphrase_success"][0])
    #print("\nNeighborhood Accuracy: ", cur_sum["neighborhood_success"][0])
