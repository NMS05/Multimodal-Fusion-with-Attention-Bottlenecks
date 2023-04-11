# Multimodal-Fusion-with-Attention-Bottlenecks

This repo provides a simple PyTorch implementation of the paper [Attention Bottlenecks for Multimodal Fusion](https://proceedings.neurips.cc/paper/2021/hash/76ba9f564ebbc35b1014ac498fafadd0-Abstract.html). This is quite different from the originally proposed architecture wherein the audio backbone is an [AST](https://arxiv.org/abs/2104.01778) pretrained on AudioSet and the visual backbone is a ViT-B16 pretained on ImageNet21k. This model is suitable for Audio-Visual classification tasks and it can be trained in the following two ways.

![MBT](https://github.com/NMS05/Multimodal-Fusion-with-Attention-Bottlenecks/blob/main/imgs/MBT.png)

(a) ***Full Fine Tuning*** - The complete model is finetuned on a downstream task.

(b) ***[AdaptFormer](https://arxiv.org/abs/2205.13535) (Recommended)*** - Perform Parameter Efficient Transfer (PET) learning using AdaptFormer, which is way more compute efficient.

![AdaptFormer](https://github.com/NMS05/Multimodal-Fusion-with-Attention-Bottlenecks/blob/main/imgs/AF.jpg)
