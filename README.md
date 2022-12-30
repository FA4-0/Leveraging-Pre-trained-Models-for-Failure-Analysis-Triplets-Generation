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

------------------------------

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
