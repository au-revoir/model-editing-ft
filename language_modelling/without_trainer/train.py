import argparse
import torch
from transformers import AutoModelForCausalLM

def load_model(model_name):
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, cache_dir="/data/local/gg676/pretrained_models")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    #--alpha_loss 0.0 --data_size 0 --prompt_rewrite True --prompt_paraphrase False --generated_paraphrase True --prompt_neighborhood True --prompt_neighborhood_type similar_examples
    parser.add_argument("--alpha_loss", type=float, required=True)
    parser.add_argument("--prompt_rewrite", action="store_true")
    parser.add_argument("--prompt_paraphrase", action="store_true")
    parser.add_argument("--generated_paraphrase", action="store_true")
    parser.add_argument("--prompt_neighborhood", action="store_true")
    parser.add_argument("--prompt_neighborhood_type", type=str, required=True, help="similar_examples, original_examples")
    args = parser.parse_args()

    main(args)
