# model_edit_
DPO + LM

```
accelerate launch --main_process_port 21688 --config_file ~/.cache/huggingface/accelerate/default_config.yaml dpo_GPTJ_with_neighbor_prompts_full_train.py --data_path data/counterfact_modifiedRandomAndSimilarWithGeneratedICLNeighborhoodPromptsWithRespectiveGoldAnswers_10000_20NeighborhoodPrompts_UPDATED_QMarkRemoved_ICLParaphrasesStarling.json --alpha_loss 0.8 --data_size 0 --prompt_rewrite True --prompt_paraphrase_type gDDenerated_prepended_examples --prompt_neighborhood True --prompt_neighborhood_type similar_examples --num_neighborhood_prompts -1 --num_epochs 6 --per_device_train_batch_size 32 --learning_rate 5e-5 --model_name EleutherAI/gpt-j-6B --dataset_name counterfact --beta 1.0
```
