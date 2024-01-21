import argparse
import os
import torch
import torch.multiprocessing as mp

from torch.distributed import init_process_group, destroy_process_group
from distributed_data_parallelism import InferenceDDP
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from utils import load_json, create_numbered_directory

bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_type=torch.bfloat16)

def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "13333"

    init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def main(rank, world_size, model_name, data_path, output_save_path, seed, icl_type):
    ddp_setup(rank, world_size)
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir="/data/local/gg676/pretrained_models", quantization_config=bnb_config, attn_implementation="flash_attention_2")#load_in_4bit=True, use_flash_attention_2=True)#torch_dtype=torch.float16)#load_in_8bit=True)#quantization_config=bnb_config) 
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="/data/local/gg676/pretrained_models")

    data = load_json(data_path)
    inferenceDDP = InferenceDDP(rank, model, tokenizer, output_save_path, max_generated_token_length=1024)
    inferenceDDP.evaluate(data, icl_type)
    destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--model_name", type=str, required=True, help="meta-llama/Llama-2-70b-chat-hf")
    #parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--output_save_path", type=str, required=True)
    parser.add_argument("--icl_type", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    world_size = torch.cuda.device_count()
    output_save_path = create_numbered_directory(args.output_save_path, args.model_name)
    mp.spawn(main, args=(world_size, args.model_name, args.data_path, output_save_path, args.seed, args.icl_type), nprocs=world_size)
