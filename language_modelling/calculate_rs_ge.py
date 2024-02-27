import argparse
import collections
import json
import numpy as np
import nltk
from tqdm import tqdm
import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer
from generate import generate_fast
from attr_snippets import AttributeSnippets
from tfidf_stats import get_tfidf_vectorizer


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
        reset_seed=False
    )

    #print("gen_texts: ", gen_texts)
    #quit()

    ngram_entropy = n_gram_entropy(gen_texts)
    consistency_tfidf = tfidf_similarity(
        " ".join(gen_texts), " ".join(consistency_texts), vec
    )

    #print("consistency_tfidf: ", consistency_tfidf)
    #quit()
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
    """
    Computes perplexity of a piece of text, measured on a reference model.
    Text is truncated to max_input_length tokens.
    """

    inputs = tok(
        [text], return_tensors="pt", max_length=max_input_length, truncation=True
    ).to("cuda")

    logits = torch.nn.functional.log_softmax(model(**inputs).logits, dim=2)
    log_probs = torch.gather(logits[:, :-1, :], 2, inputs["input_ids"][:, 1:, None])[0]

    # Perplexity = exp(-1/N * log P(x_1, ..., x_n))
    return torch.exp(-1 / inputs["input_ids"].size(1) * log_probs.sum()).item()

def main(args):
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    #model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float16, cache_dir="/data/local/gg676/pretrained_models").to("cuda")
    #model = AutoPeftModelForCausalLM.from_pretrained("/common/home/gg676/ModelEditing/DPO/FTLayer21SheetRow12/checkpoint-1000", torch_dtype=torch.float16, cache_dir="/data/local/gg676/pretrained_models").to("cuda")
    model_path = "/common/home/gg676/ModelEditing/DPO/FTMaskWith10RandomWikiForConsistencyTextFactor0.1MaxTokens256/checkpoint-16878"
    print("Model path: ", model_path)
    model = AutoPeftModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, cache_dir="/data/local/gg676/pretrained_models").to("cuda")

    #model = AutoPeftModelForCausalLM.from_pretrained("/common/home/gg676/ModelEditing/DPO/gptJ_DPO_PROMPTMASK_allAugmentationsRandom.pt", torch_dtype=torch.float16, cache_dir="/data/local/gg676/pretrained_models").to("cuda")
    #model.load_state_dict(torch.load("/common/home/gg676/ModelEditing/DPO/FTLayer21WithoutLora_SheetRow11/layer21_withoutLoRA..pth"))
    #model = AutoPeftModelForCausalLM.from_pretrained("/common/home/gg676/ModelEditing/DPO/gpt-j-6B_alpha1.0_neigbhorhoodTypeno_examples.pt", torch_dtype=torch.float16, cache_dir="/data/local/gg676/pretrained_models").to("cuda")
    #model = AutoModelForCausalLM.from_pretrained("/data/local/gg676/model_editing/saved_models/gpt2-xl_alpha1.0_neigbhorhoodTypesimilar_examples.pt", torch_dtype=torch.float16, cache_dir="/data/local/gg676/pretrained_models").to("cuda")
    original_data = data = load_json(args.original_data_path)
    eval_data = original_data[:10000]
    snips = AttributeSnippets("data/attribute_snippets/attribute_snippets.json")
    vec = get_tfidf_vectorizer("data/tfidf")
    result_list = []
    cur_sum = collections.defaultdict(lambda: [])
    start = 1250 * args.start_multiplier
    for record in tqdm(eval_data[start: start+args.start_end_delta]):
        ret = {}

        subject, target_new, target_true = (
        record["requested_rewrite"][x] for x in ["subject", "target_new", "target_true"]
    )
        generation_prompts = record["generation_prompts"]
        rel_id = record["requested_rewrite"]["relation_id"]
        consistency_texts = [x["text"] for x in snips[rel_id][target_new["id"]]]
        #print("generation_prompts: ", generation_prompts)
        #print("consistency_texts: ", consistency_texts[0])
        #print("len(consistency_texts): ", len(consistency_texts))
        #quit()
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
        #print("ret: ", ret)
        result_list.append(ret)
        #quit()
    for data in result_list:
        for key in ["ngram_entropy", "reference_score", "essence_score"]:
            if key in data:
               cur_sum[f"{key}"].append(data[key])
    cur_sum = {k: (np.mean(v), np.std(v)) for k, v in cur_sum.items()}
    for k, v in cur_sum.items():
        if all(exclude not in k for exclude in ["essence_score", "time"]):
            # Constant multiplication scales linearly with mean and stddev
            cur_sum[k] = tuple(np.around(z * 100, 2) for z in v)
    #print("before cur_sum: ", cur_sum)
    print("cur_sum: ", cur_sum)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--original_data_path", type=str, required=True)
    #parser.add_argument(--model_saved_path, type=str, help="lora saved path")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--start_multiplier", type=int, required=True)
    parser.add_argument("--start_end_delta", type=int, required=True)
    args = parser.parse_args()
    main(args)    
