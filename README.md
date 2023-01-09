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

|   Model   | BLEU-1 | BLEU-3 |  MET. | ROUGE-1 |       |       | ROUGE-L |       |       | LESE-1 |       |       | Lev-1 | LESE-3 |       |      | Lev-3 |  PPL |
|:---------:|:------:|:------:|:-----:|:-------:|:-----:|:-----:|:-------:|:-----:|:-----:|:------:|:-----:|:-----:|:-----:|:------:|:-----:|:----:|:-----:|:----:|
|     ~     |    ~   |    ~   |   ~   |  Prec.  |  Rec. |   F1  |  Prec.  |  Rec. |   F1  |  Prec. |  Rec. |   F1  |   ~   |  Prec. |  Rec. |  F1  |   ~   |   ~  |
|    BART   |  3.19  |  1.69  |   -   |    -    |   -   |   -   |    -    |   -   |   -   |  2.35  |  4.29 |  2.55 |  72.0 |  0.01  |  0.0  |  0.0 |  24.0 |  1.0 |
|    BERT   |  1.86  |  0.67  |  4.04 |   6.85  | 11.95 |  8.09 |   5.74  | 10.16 |  6.8  |  1.41  | 10.38 |  2.36 | 286.0 |  0.01  |  0.02 | 0.01 |  96.0 |  1.0 |
|  ROBERTA  |  0.08  |  0.07  |  0.24 |   0.14  |  0.37 |  0.19 |   0.14  |  0.36 |  0.19 |  0.06  |  0.22 |  0.09 | 194.0 |   0.0  |  0.0  |  0.0 |  64.0 |  1.0 |
|    GPT3   |  10.94 |  7.55  | 32.07 |  16.64  |  52.9 | 24.31 |  14.53  | 46.98 | 21.31 |  8.66  | 46.48 | 14.04 | 187.0 |  3.75  | 20.68 | 6.11 |  65.0 | 1.39 |
|   GPT2-B  |  20.83 |  15.13 | 25.87 |   30.5  | 32.33 | 28.89 |  27.79  | 29.51 | 26.32 |  20.99 | 21.26 | 19.04 |  42.0 |  9.04  |  9.31 | 8.27 |  15.0 | 1.42 |
|   GPT2-M  |  21.19 |  15.45 | 26.32 |  30.91  | 32.53 |  29.1 |  28.35  | 29.81 | 26.65 |  21.7  | 21.82 | 19.67 |  42.00 |   9.60  |  9.65 | 8.69 |  15.0 | 1.52 |
|   GPT2-L  |  22.38 |  16.59 | 28.49 |  30.69  | 34.99 | 30.16 |  28.28  | 32.23 | 27.76 |  21.62 | 23.87 | 20.56 |  43.0 |  10.06 | 11.13 | 9.59 |  16.0 | 1.42 |
|   W-BART  |  3.37  |  1.81  |  3.36 |   5.37  |  7.69 |  5.23 |   4.54  |  6.11 |  4.27 |  2.66  |  3.40  |  2.37 |  63.0 |  0.14  |  0.01 | 0.02 |  21.0 |  1.0 |
|   W-BERT  |  0.23  |  0.16  |  0.46 |   0.81  |  1.08 |  0.81 |   0.77  |  1.03 |  0.77 |   0.2  |  0.62 |  0.21 | 179.0 |  0.01  |  0.0  |  0.0 |  59.0 |  1.0 |
| W-ROBERTA |  0.06  |  0.05  |  0.12 |   0.08  |  0.15 |  0.1  |   0.08  |  0.14 |  0.1  |  0.03  |  0.07 |  0.04 |  89.0 |   0.0  |  0.0  |  0.0 |  29.0 |  1.0 |
|   W-GPT3  |  11.01 |  7.53  |  31.80 |  16.19  | 52.43 | 23.75 |  13.99  | 46.08 |  20.6 |  8.62  | 45.66 | 13.94 | 185.0 |  3.67  | 19.97 | 5.96 |  64.0 | 1.27 |
|  W-GPT2-B |  21.20  |  15.39 | 26.46 |  30.23  | 32.89 | 29.05 |  27.58  | 30.05 |  26.5 |  21.03 | 21.85 |  19.4 |  42.0 |  9.09  |  9.53 | 8.42 |  15.0 | 1.29 |
|  W-GPT2-M |  21.91 |  16.11 | 27.17 |  31.61  | 33.62 | 30.07 |  28.99  | 30.85 | 27.55 |  21.85 | 22.45 | 20.07 |  42.0 |  9.82  | 10.11 | 9.03 |  15.00 | 1.35 |
|  W-GPT2-L |  22.31 |  16.49 | 28.62 |  30.89  | 35.46 | 30.32 |  28.33  | 32.47 | 27.76 |  21.38 | 23.98 | 20.43 |  44.0 |  10.0  | 11.18 | 9.55 |  16.00 | 1.29 |

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
