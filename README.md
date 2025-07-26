**Ophanim** provides a streamlined framework for customizing Llama 3.2 models using contrastive learning with JSON-based training datasets. Users can upload flexible datasets containing positive and negative examples to train models to exhibit desired behaviors while avoiding unwanted ones, with built-in regularization to prevent overfitting. It leverages Llama 3.2’s tokenizer and embeddings for maximum compatibility with llama.cpp, supports GGUF model export, and allows live testing via one-shot prompts. Installation is simple with Hugging Face integration and Streamlit UI, making it easy to configure and run.

# ✨ Features

📁 **JSON Dataset Upload**: 
- Upload custom training datasets in flexible JSON format
- Use any prompt structure not limited to the Instruction-Following
- Give as many negative examples as you like for every positive example

🎯 **Contrastive Learning**: Train models from scratch to exhibit desired behaviors while avoiding unwanted patterns
- Maximizes probability of positive behavioral examples, train model how to "behave"
- Minimizes probability of negative behavioral examples, train model how to "avoid"
- Regularization prevents overfitting and ensures that power of positive and negative examples remains in a 1:1 ratio

🦙 **Llama 3.2 Foundation**: Uses Llama 3.2 1B tokenizer and embeddings as architectural skeleton
- Customize your Llama model in more detail than standard fine tuning allows
- Uses ready-made skeleton and architecture of Llama to make maximum compatibility with llama.cpp
- Use existing Llama embeddings and tokenizer to help the model better understand the language

# ⚡ And... more!

**GGUF Export**: Download your trained model in GGUF format for llama.cpp compatibility

**Live Testing**: Test your custom-trained model with one-shot prompts

**Simple Configuration**: Essential training parameters without complexity overload

# 📊 Dataset Format

Upload your training data as a JSON file with the following structure:

```
[
  {
    "positive_example": "(Positive example)",
    "negative_example_1": "(Negative example 1)",
    "negative_example_2": "(Negative example 2)"
  }
]
```

Example for standart Llama Instruction-Following prompt:

```
[
  {
    "positive_example": "<|start_header_id|>system<|end_header_id|>\n\nYou are an AI\n<|start_header_id|>user<|end_header_id|>\n\nAre you AI?\n<|start_header_id|>assistant<|end_header_id|>\n\nYes, I am.",
    "negative_example_1": "<|start_header_id|>system<|end_header_id|>\n\nYou are an AI\n<|start_header_id|>user<|end_header_id|>\n\nAre you AI?\n<|start_header_id|>assistant<|end_header_id|>\n\nNo, I am not."
  }
]
```

# 🚀 Installation & Usage
1. Download GitHub repository as archive, and unpack it in your desired folder.
2. Sign Up in Hugging Face and get access to model Llama 3.2 1B Instruct
3. Run the following command:
```
huggingface-cli login
[paste token here]
y
streamlit run C:/path/to/your/app.py
```

# 🙏 Acknowledgments

- Meta AI for Llama 3.2 model, tokenizer and embedding foundations
- Hugging Face for transformers infrastructure
- Streamlit team for the intuitive framework
- llama.cpp community for GGUF format specification

- **Aditya Verma** for help in rewriting the test code and general friendly support.
- **ChatGPT, Grok, Claude** and their developers for making this project possible.

# Looking for friendly Community?
**The CodeVerse Hub**: https://discord.gg/9Gdem4RbEf

# License
This code is generally licensed under the MIT License, with the following exceptions:
- Due to community disregard and unethical practices, this code is prohibited for use by Character.AI employees or employees of other companies that worked on Character.AI less than 4 years ago.
- You must not use this code to spread misinformation, falsehoods or defamation, or deepfakes of any kind, and you must take all reasonable care to avoid doing so.
- Any use of Meta's Llama Models in code is automatically subject to Meta's Llama Models license under local law.
