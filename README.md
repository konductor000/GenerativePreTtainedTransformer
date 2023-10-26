
## Generative Pretrained Transformer

This project implements a decoder-only Transformer model for text generation using PyTorch, with optimizations like mixed precision and flash attention for faster training and inference. The goal is to train small and large models on high-quality datasets, then apply reinforcement learning from human feedback (RLHF) to create a conversational chat model.

#### Flash attention
[Flash attention](https://arxiv.org/abs/2307.08691) improves model training throughput by 3-4x compared to regular attention. It also speeds up the model's forward pass by 10x during inference. The reduced computational and memory overhead allows training larger transformer models with higher throughput and lower cost.


### TODO
*This project is currently in an unresolved state and here is the list of tasks that still need to be completed*

&check; Implement Transformer model on PyTorch with GPU support

&check; Create dataset and training loop

&check; Create model evaluation on different metrics *BLEU, ROUGE* and add [wandb.ai](https://wandb.ai/) logging

&check; Apply optimizations *flash attention, mixed precision*

&check; Train 24M tiny model on little synthetic dataset

&check; Train larger 360M model on filtered Pile dataset

&cross; Finetune pretrained model on high quality datasets

&cross; Apply RLHF. Create reward model and train it

&cross; Finetune pretrained model using instruct datasets and reward model

&cross; Multi-GPU training support

&cross; Create UI/API for chat model

### Tiny model
Trhis model is a tiny model trained on a high quality synthetic [dataset](https://huggingface.co/datasets/roneneldan/TinyStories) consisting of 400M tokens from short stories for children generated using GPT-3.5 and GPT-4. It has a small number of unique tokens. The model was trained for 6 epochs (around 15 hours) on an RTX 3090 GPU. The test loss reached 1.15 on the CrossEntropyLoss.

#### Model Details
- Embedding size: 256
- Max sequence length: 256
- Number of transformer blocks: 12
- Number of heads: 8
- Batch size: 178
- Number of trainable parameters: 24M
### Large model
The current model is a large model trained on a filtered pile [dataset](https://huggingface.co/datasets/stanford-crfm/DSIR-filtered-pile-50M) consisting of high quality sequences filtered using DSIR with target datasets Wikipedia, books. The model was trained on 7B tokens on RTX 4090 GPU (around 40 hours). The test loss reached 2 on the CrossEntropyLoss.

#### Model Details
- Embedding size: 1024
- Max sequence length: 512
- Number of transformer blocks: 24
- Number of heads: 16
- Batch size: 30
- Number of trainable parameters: 360M


