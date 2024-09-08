import argparse
import collections
import json
import numpy as np
import nltk
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from generate import generate_fast

def load_json(path):
    with open(path) as fp:
        data = json.load(fp)
    return data

def test_generation(
    model,
    tok,
    prefixes,
    consistency_texts,
    essence_texts,
    vec
):
    gen_texts = generate_fast(
        model,
        tok,
        prefixes,
        n_gen_per_prompt=1,
        max_out_len=100,
    )

    ngram_entropy = n_gram_entropy(gen_texts)
    consistency_tfidf = tfidf_similarity(
        " ".join(gen_texts), " ".join(consistency_texts), vec
    )

    ret = {
        "ngram_entropy": ngram_entropy,
        "reference_score": consistency_tfidf,
        "text": gen_texts,
    }

    if len(essence_texts) > 0:
        ppl = perplexity(model, tok, " ".join(essence_texts), max_input_length=100)
        ret.update({"essence_score": ppl, "essence_text": essence_texts})

    return ret


def n_gram_entropy(gen_texts, agg="arith"):
    assert agg in ["arith", "geom"]

    return (scipy.stats.mstats.gmean if agg == "geom" else np.mean)(
        [compute_n_gram_entropy(txt) for txt in gen_texts]
    ).item()


def compute_n_gram_entropy(sentence, ns=None, weights=None, agg="arith"):
    if ns is None:
        ns = [2, 3]
    if weights is None:
        weights = [2 / 3, 4 / 3]
    assert agg in ["arith", "geom"]

    entropy_list = []
    for n in ns:
        fdist = compute_freq(sentence, n)
        freqs = np.array([freq for _, freq in fdist.items()])
        freqs = freqs / freqs.sum()

        entropy_list.append(np.sum(-freqs * np.log(freqs) / np.log(2)))

    entropy_list = np.array(entropy_list) * np.array(weights)

    return (scipy.stats.mstats.gmean if agg == "geom" else np.mean)(entropy_list)


def compute_freq(sentence, n=2):
    tokens = nltk.word_tokenize(sentence)
    ngrams = nltk.ngrams(tokens, n)
    return nltk.FreqDist(ngrams)


def tfidf_similarity(text_a, text_b, vec):
    encs = vec.transform([text_a, text_b]).A
    norm = np.linalg.norm
    return (np.dot(encs[0], encs[1]) / norm(encs[0]) / norm(encs[1])).item()

def perplexity(
    model,
    tok,
    text,
    max_input_length=None,
):
    inputs = tok(
        [text], return_tensors="pt", max_length=max_input_length, truncation=True
    ).to(model.device)

    logits = torch.nn.functional.log_softmax(model(**inputs).logits, dim=2)
    log_probs = torch.gather(logits[:, :-1, :], 2, inputs["input_ids"][:, 1:, None])[0]

    return torch.exp(-1 / inputs["input_ids"].size(1) * log_probs.sum()).item()

def perform_generation_test(model, tokenizer, record, snips, vec): 
    
    result_list = []
    cur_sum = collections.defaultdict(lambda: [])

    ret = {}

    subject, target_new, target_true = (
    record["requested_rewrite"][x] for x in ["subject", "target_new", "target_true"]
)
    generation_prompts = record["generation_prompts"]
    rel_id = record["requested_rewrite"]["relation_id"]
    consistency_texts = [x["text"] for x in snips[rel_id][target_new["id"]]]
    essence_texts = [
        x["text"]
        for x in snips[rel_id][target_new["id"]]
        if x["name"] == record["requested_rewrite"]["subject"]
    ]
    assert (
        len(consistency_texts) > 0
    ), "Must have consistency texts to evaluate generation"
    gen_stats = test_generation(
        model,
        tokenizer,
        generation_prompts,
        consistency_texts,
        essence_texts,
        vec,
    )
    ret.update(gen_stats)
    result_list.append(ret)
    for data in result_list:
        for key in ["ngram_entropy", "reference_score"]:
            if key in data:
               cur_sum[f"{key}"].append(np.around(data[key] * 100, 2))
    return cur_sum

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--original_data_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--start_multiplier", type=int, required=True)
    parser.add_argument("--start_end_delta", type=int, required=True)
    args = parser.parse_args()
    main(args)    
