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

> - `git clone https://github.com/AI-for-Fault-Analysis-FA4-0/Leveraging-Pre-trained-Models-for-Failure-Analysis-Triplets-Generation`: clone repository
> - ...

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
