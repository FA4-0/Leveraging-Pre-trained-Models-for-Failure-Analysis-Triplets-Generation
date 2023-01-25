# Leveraging-Pre-trained-Models-for-Failure-Analysis-Triplets-Generation

### ABSTRACT
```
Pre-trained Language Models recently gained traction in the Natural Language Processing (NLP)
domain for text summarization, generation and question answering tasks. This stems from the
innovation introduced in Transformer models and their overwhelming performance compared with
Recurrent Neural Network Models (Long Short Term Memory (LSTM)). In this paper, we leverage
the attention mechanism of pre-trained causal language models such as Transformer model for
the downstream task of generating Failure Analysis Triplets (FATs) - a sequence of steps for
analyzing defected components in the semiconductor industry. We compare different transformer
model for this generative task and observe that Generative Pre-trained Transformer 2 (GPT2)
outperformed other transformer model for the failure analysis triplet generation (FATG) task.
In particular, we observe that GPT2 (trained on 1.5B parameters) outperforms pre-trained 
BERT, BART and GPT3 by a large margin on ROUGE. Furthermore, we introduce LEvenshstein
Sequential Evaluation metric (LESE) for better evaluation of the structured FAT data and
show that it compares exactly with human judgment than existing metrics.
```

------------------------------

### How to use

 - Clone repository: ```git clone https://github.com/AI-for-Fault-Analysis-FA4-0/Leveraging-Pre-trained-Models-for-Failure-Analysis-Triplets-Generation```
 - Runing training and evaluation example
    - ```python
           python pretrainer.py \
           --model_type gpt2 \
           --model_name_or_path gpt2 \
           --do_train \
           --do_eval \
           --max_seq_length 128 \
           --per_gpu_train_batch_size 1 \
           --learning_rate 5e-5 \
           --num_train_epochs 5.0 \
           --output_dir result/ \
           --eval_dir evaluation/ \
           --overwrite_output_dir \
           --fp16 \
           --fp16_opt_level O2 \
           --gradient_accumulation_steps 1 \
           --seed 42 \
           --do_lower_case \
           --warmup_steps 100 \
           --logging_steps 100 \
           --save_steps 100 \
           --evaluate_during_training \
           --save_total_limit 1 \
           --adam_epsilon 1e-8 \
           --weight_decay 0.05 \
           --max_grad_norm 1.0 \
           --return_token_type_ids \
           #--use_weights \
           --max_steps -1
           ```
- Model type/name with Causal LMHead
  - ```facebook/bart-large-cnn```: Bidirectional Auto-Regressive Transformer
  - ```bert-base-uncased```: Bidirectional Encoder Representations from Transformers
  - ```roberta-large```: Robustly Optimized BERT Pretraining Approach
  - ```distilbert-base-uncased```: A distilled version of BERT: smaller, faster, cheaper and lighter
  - ```xlnet-large-cased```: Generalized Autoregressive Pretraining for Language Understanding
  - ```openai-gpt```: Generative Pre-trained Transformer 3
  - ```gpt2```: Generative Pre-trained Transformer 2 (base)
  - ```gpt2-medium```: Generative Pre-trained Transformer 2 (Medium)
  - ```gpt2-large```: Generative Pre-trained Transformer 2 (Large)

------------------------------

### Results

| Model     | BLEU-1 | BLEU-3 | MET.  | ROUGE-1 |  |  |ROUGE-L  |  |  | LESE-1   |       |       |   Lev-1    |   LESE-3    |       |      |   Lev-3   |        PPL            |
|-----------|--------|--------|-------|---------|---------|--------|-------|--------|-------|-------|-------|-------|-------|-------|-------|------|------|--------------------|
|           |        |        |       | Prec.   | Rec.    | F1     | Prec. | Rec.   | F1    | Prec. | Rec.  | F1    |       | Prec. | Rec.  | F1   |      |                    |
| BART      | 3.04   | 1.67   | 3.23  | -       | -       | -      | -     | -      | -     | 2.12  | 3.85  | 2.28  | 73.0  | 0.01  | 0.0   | 0.0  | 24.0 | 1.0                |
| BERT      | 1.81   | 0.65   | 4.0   | 6.7     | 12.0    | 7.99   | 5.62  | 10.21  | 6.72  | 1.38  | 10.48 | 2.33  | 287.0 | 0.02  | 0.03  | 0.01 | 96.0 | 1.0                |
| ROBERTA   | 0.14   | 0.11   | 0.32  | 0.26    | 0.56    | 0.34   | 0.26  | 0.55   | 0.34  | 0.09  | 0.33  | 0.13  | 169.0 | 0.0   | 0.0   | 0.0  | 56.0 | 1.0                |
| GPT3      | 22.71  | 16.6   | 29.34 | 30.06   | 35.75   | 30.26  | 27.65 | 32.93  | 27.83 | 20.88 | 24.93 | 20.64 | 45.0  | 9.34  | 11.28 | 9.29 | 16.0 | 1.53 |
| GPT2-B    | 20.85  | 15.25  | 25.67 | 30.66   | 31.86   | 28.78  | 28.1  | 29.2   | 26.35 | 21.31 | 21.1  | 19.17 | 41.0  | 9.31  | 9.3   | 8.39 | 15.0 | 1.42 |
| GPT2-M    | 21.26  | 15.47  | 26.74 | 30.37   | 33.28   | 29.15  | 27.65 | 30.4   | 26.56 | 21.08 | 22.06 | 19.41 | 43.0  | 9.23  | 9.79  | 8.55 | 15.0 | 1.52 |
| GPT2-L    | 22.87  | 16.87  | 28.7  | 31.88   | 35.19   | 30.87  | 29.19 | 32.24  | 28.24 | 22.01 | 23.83 | 20.81 | 42.0  | 10.06 | 10.89 | 9.53 | 15.0 | 1.412 |
| W_BART    | 4.71   | 1.87   | 5.04  | 5.41    | 11.05   | 6.57   | 4.36  | 8.97   | 5.28  | 2.56  | 5.9   | 3.17  | 81.0  | 0.0   | 0.0   | 0.0  | 27.0 | 1.01 |
| W_BERT    | 0.42   | 0.2    | 0.56  | 1.08    | 1.49    | 1.15   | 0.97  | 1.37   | 1.04  | 0.33  | 1.15  | 0.44  | 74.0  | 0.01  | 0.0   | 0.0  | 24.0 | 1.0                |
| W_ROBERTA | 0.07   | 0.06   | 0.33  | 0.11    | 0.32    | 0.16   | 0.11  | 0.31   | 0.15  | 0.05  | 0.2   | 0.07  | 196.0 | 0.0   | 0.0   | 0.0  | 65.0 | 1.0                |
| W_GPT3    | -      | -      | -     | -       | -       | -      | -     | -      | -     | -     | -     | -     | -     | -     | -     | -    | -    | 1.34  |
| W_GPT2-B  | 21.15  | 15.52  | 26.21 | 31.27   | 32.49   | 29.33  | 28.51 | 29.64  | 26.72 | 21.64 | 21.52 | 19.49 | 41.0  | 9.59  | 9.62  | 8.65 | 15.0 | 1.28 |
| W_GPT2-M  | 20.99  | 15.38  | 26.34 | 30.05   | 32.53   | 28.68  | 27.44 | 29.76  | 26.2  | 21.0  | 21.68 | 19.28 | 42.0  | 9.43  | 9.75  | 8.67 | 15.0 | 1.34  |
| W_GPT2-L  | 22.67  | 16.64  | 28.68 | 31.66   | 35.54   | 30.89  | 29.02 | 32.54  | 28.26 | 21.93 | 24.09 | 20.8  | 43.0  | 10.1  | 11.17 | 9.62 | 16.0 | 1.26 |


### Cite

```
@misc{https://doi.org/10.48550/arxiv.2210.17497,
  doi = {10.48550/ARXIV.2210.17497},
  url = {https://arxiv.org/abs/2210.17497},
  author = {Ezukwoke, Kenneth and Hoayek, Anis and Batton-Hubert, Mireille and Boucher, Xavier and Gounet, Pascal and Adrian, Jerome},
  keywords = {Computation and Language (cs.CL), Machine Learning (cs.LG), Applications (stat.AP), FOS: Computer and information sciences, FOS: Computer and information sciences, G.3; I.2; I.7, 68Txx, 68Uxx},
  title = {Leveraging Pre-trained Models for Failure Analysis Triplets Generation},
  publisher = {arXiv},
  year = {2022},
  copyright = {Creative Commons Attribution 4.0 International}
}
```
