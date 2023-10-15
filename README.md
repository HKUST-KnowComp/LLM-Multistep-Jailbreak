# Multi-step Jailbreaking Privacy Attacks on ChatGPT
Paper Link: https://arxiv.org/pdf/2304.05197.pdf

Code for Findings-EMNLP 2023 paper: Multi-step Jailbreaking Privacy Attacks on ChatGPT.

### Disclaimer 
Currently, our proposed prompt attacks are no longer valid for OpenAI models. If you would like to reproduce our experimental results, you need to run our attacks on earlier versions such as ```gpt-3.5-turbo-0301```.

Additionally, if you find our web-sourced data includes your personal information and would like to remove it, please send us an email and we will remove your data records immediately.

### PII Extraction Attacks
1. Properly set up the API and path in ```config.py```. You need to specify the paths to save the extraction results.

2. Supported attack templates:

```- DQ:``` Direct query to extract PII.

```- JQ:``` Query with jailbreak template to extract PII.

```- JQ+COT:``` Query with pre-defined multi-step context and jailbreak template to extract PII.

```- JQ+MC:``` Query with jailbreak template to extract PII for multiple responses and use a multi-choice template to let LLM select the final answer.

```- JQ+COT+MC:``` Query with pre-defined multi-step context and jailbreak template to extract PII for multiple responses and use a multi-choice template to let LLM select the final answer.


#### Run Attacks on OpenAI APIs

run ```chatgpt_extraction.py```.
#### Run Attacks on open-source LLMs

run ```llm_extraction.py```.
#### Attacks on Email Content Recovery
To reproduce our attacks to extract email content, you may refer to the 
[email_content_extraction](https://github.com/HKUST-KnowComp/LLM-Multistep-Jailbreak/tree/main/email_content_extraction) folder.


#### Result Analysis

First, obtain ```.csv``` paths after running the attacks.

Second, specify the result paths in ```config.py``` or ```pred_analysis.py```.

Then, run ```pred_analysis.py```.
### Citation

```bibtex
@inproceedings{Li2023MultistepJP,
  author       = {Haoran Li and Dadi Guo and Wei Fan and Mingshi Xu and Jie Huang and Yangqiu Song},
  title        = {Multi-step Jailbreaking Privacy Attacks on ChatGPT},
  booktitle    = {Findings of the Association for Computational Linguistics: EMNLP 2023},
  month        = {dec},
  year         = {2023},
  url          = {https://arxiv.org/abs/2304.05197},
}
```

### Miscellaneous

Please send any questions about the code and/or the algorithm to hlibt@connect.ust.hk
