# References and Resources

## Target Role

**Apple — Generative AI Applied Scientist, SIML · ISE (Role 200632699-0836)**
https://jobs.apple.com/en-us/details/200632699-0836/generative-ai-applied-scientist-siml-ise?team=MLAI

---

## Datasets

### GQA (primary static-image benchmark)
- About: https://cs.stanford.edu/people/dorarad/gqa/about.html
- Paper: Hudson & Manning, *GQA: A New Dataset for Real-World Visual Reasoning and Compositional Question Answering.* https://arxiv.org/abs/1902.09506

Real-world images, cleaned scene graphs (objects/attributes/relations), compositional questions, functional programs, and evaluation ideas including grounding and consistency.

### Visual Genome (supporting annotation source)
- API / downloads: https://homes.cs.washington.edu/~ranjay/visualgenome/api.html
- Paper: Krishna et al., *Visual Genome: Connecting Language and Vision Using Crowdsourced Dense Image Annotations.* https://arxiv.org/abs/1602.07332

### CLEVR (controlled diagnostic, ~30 examples)
- Site: https://cs.stanford.edu/people/jcjohns/clevr/

Synthetic scenes with ground-truth locations, attributes, relationships, questions, answers, and functional programs. Diagnostic appendix only.

### COCO (optional generalization / sanity)
- Site: https://cocodataset.org/

---

## Models

### Qwen2.5-VL
- Transformers docs: https://huggingface.co/docs/transformers/en/model_doc/qwen2_5_vl
- Model card: https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct
- Technical report: https://arxiv.org/abs/2502.13923
- Blog: https://qwenlm.github.io/blog/qwen2.5-vl/

### LLaVA-NeXT
- GitHub: https://github.com/LLaVA-VL/LLaVA-NeXT
- Overview: https://llava-vl.github.io/blog/2024-01-30-llava-next/
- LLaVA-NeXT-Interleave paper: https://arxiv.org/abs/2407.07895

---

## Multimodal Inference
- Image-text-to-text guide: https://huggingface.co/docs/transformers/en/tasks/image_text_to_text
- Transformers docs: https://huggingface.co/docs/transformers/en/index

---

## License note
Code in this repository is MIT-licensed. Datasets and model weights are governed
by their own licenses — review and comply with each before use.
