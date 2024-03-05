# Setup Instructions

```
conda create --name model_edit python=3.9.11
conda activate model_edit
pip install -r requirements.txt
```
Download `attribute_snippets.json`, `multi_counterfact.json`, `idf.npy` and `tfidf_vocab.json` from [here](https://rome.baulab.info/data/dsets/).
Place `attribute_snippets.json` under `data/attribute_snippets`, `multi_counterfact.json` under `data/counterfact`.
Place `idf.npy` and `tfidf_vocab.json` under `data/tfidf`.

# Commands

## Mass Edit
### Counterfact
To train and calculate the ES, PS and NS after including the edit, augmented paraphrase and neighborhood prompts run:
```
accelerate launch run.py --data_path data/counterfact/counterfact_mass_edit_augmented.json --alpha_loss 1.0 --prompt_rewrite True --prompt_paraphrase_type generated_prepended_examples --generated_prepended_words_path data/generated_prepended_words_paraphrases.json --prompt_neighborhood_type random_examples --num_epochs 6 --per_device_train_batch_size 32 --learning_rate 7e-5 --model_name EleutherAI/gpt-j-6B --dataset_name counterfact --beta 0.5 --mask_prompt True --output_dir saved_models/ --include_consistency_text False
```
`beta` value will only be used for DPO which is enabled when `alpha != 1`. Setting `alpha = 0` will result in pure DPO training.
Setting either `prompt_paraphrase_type` and/or `prompt_neighborhood_type`to `no_examples` will exclude the augmented paraphrases and neighborhood prompts from train set used for training.

To calculate the consistency and fluency scores for each of the example, run
```
bash calculate_ge_rs.sh 8 saved_models/checkpoint EleutherAI/gpt-j-6B
```
Replace the first argument `8` with the number of GPUs you can use. The second argument is the path of the LoRA adapter that was saved after the training above.
The results will be stored under `results/run_xx`

To get the final consistency and fluency score of all the examples run
```
python summarize_consistency_fluency.py --saved_result_path results/run_xx
```

### zSRE
Similarly to train and calculate the ES, PS and NS run:
```
accelerate launch run.py --data_path data/zsre/zsre_eval.json --alpha_loss 1.0 --prompt_rewrite True --prompt_paraphrase_type generated_prepended_examples --generated_prepended_words_path data/generated_prepended_words_paraphrases.json --prompt_neighborhood_type random_examples --num_epochs 6 --per_device_train_batch_size 32 --learning_rate 7e-5 --model_name EleutherAI/gpt-j-6B --dataset_name zsre --beta 0.5 --mask_prompt True --output_dir saved_models/ --include_consistency_text False
```
zSRE does not have consistency and fluency metrics.
