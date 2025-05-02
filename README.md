# Python Code Generator with SmolLM

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.45+-red.svg)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)
![SmolLM](https://img.shields.io/badge/SmolLM-140-brightgreen.svg)
![uv](https://img.shields.io/badge/uv-package_manager-purple.svg)

A professional Streamlit application that generates Python code from natural language descriptions using a fine-tuned SmolLM language model. This application demonstrates advanced model deployment techniques and proper software engineering practices for AI-based code generation tools.

![App Screenshot](https://github.com/user-attachments/assets/83f08bad-0594-49fd-99ed-47a7e5ebbfe0)


## 🌟 Features

- **AI-Powered Code Generation**: Converts natural language descriptions to functional Python code
- **Interactive Code Editor**: Built-in syntax highlighting with streamlit-ace
- **Code Execution**: Sandboxed environment for testing generated code
- **Dependency Management**: Install required Python packages on-the-fly
- **Error Handling**: Comprehensive error capture and display
- **Responsive Design**: Clean, modern UI with optimized layout
- **Session State Management**: Persistent code storage between interactions
- **Robust Logging**: Detailed logging for debugging and monitoring

## 🏗️ Technical Architecture

The application follows a modular, object-oriented architecture with clear separation of concerns:

### Core Components

1. **ModelLoader Class**: 
   - Handles model initialization, caching, and inference
   - Implements hardware acceleration detection (CUDA, MPS, CPU)
   - Manages tokenization and text generation

2. **CodeExecutor Class**:
   - Provides secure code execution in a sandboxed environment
   - Captures stdout/stderr and exception information
   - Handles dependency installation via subprocess

3. **AppUI Class**:
   - Manages Streamlit UI components and layouts
   - Handles session state initialization and management
   - Controls rendering and UI events

4. **Main Application Flow**:
   - Structured with proper error handling and logging
   - Uses a clean dependency injection pattern

## 🤖 Model Access

The fine-tuned model is available on the Hugging Face Hub:

- **Model:** [LordSahu/smollm-codesearchnet-sft-peft-finetuned_v1](https://huggingface.co/LordSahu/smollm-codesearchnet-sft-peft-finetuned_v1)

You can easily use this model in your own projects:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model_id = "LordSahu/smollm-codesearchnet-sft-peft-finetuned_v1"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
prompt = "Write a function to calculate the factorial of a number"
result = pipe(prompt, max_new_tokens=400, temperature=0.1, top_p=0.95)
print(result[0]['generated_text'])
```

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher
- [uv](https://docs.astral.sh/uv/) - Fast Python package installer and environment manager

### Installation

1. Clone this repository:
```bash
git clone https://github.com/utkarsh-iitbhu/slm-code-sft-peft.git
cd slm-code-sft-peft
```

2. Install uv (if not already installed):
```bash
# On macOS with Homebrew
brew install uv

# On macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

3. Set up the environment and install dependencies:
```bash
uv sync
```

4. Run the application:
```bash
streamlit run app.py
```

## 🔧 Key Dependencies

- **streamlit**: Web application framework
- **streamlit-ace**: Code editor with syntax highlighting
- **transformers**: HuggingFace Transformers library
- **peft**: Parameter-Efficient Fine-Tuning library
- **torch**: PyTorch for model inference
- **logging**: Standard library for application logging

## 💻 How to Use

1. **Generate Code**:
   - Enter a natural language description of the code you need
   - Adjust temperature slider (higher = more creative, lower = more deterministic)
   - Set maximum token length for output
   - Click "Generate Code" button

2. **Edit and Run**:
   - Review generated code in the editor
   - Make any necessary adjustments
   - Click "Run Code" to execute and see results

3. **Manage Dependencies**:
   - Use the "Need to install packages?" section to add required libraries

## 🧠 Model Details

The application uses a fine-tuned version of the SmolLM model:

- **Base Model**: HuggingFaceTB/SmolLM-135M (135 million parameters)
- **Fine-Tuning Method**: Low-Rank Adaptation (LoRA)
- **Training Data**: Python code examples from CodeSearchNet
- **Training Technique**: Supervised Fine-Tuning with docstring-to-code mapping
- **Checkpoint**: The application uses checkpoint from the fine-tuned model

### About SmolLM

SmolLM is a family of compact language models available in three sizes: 135M, 360M, and 1.7B parameters. They have been trained on a high-quality corpus (SmolLM-Corpus) which includes:

- **Cosmopedia v2**: Synthetic textbooks and educational content
- **FineWeb-Edu**: Educational web content
- **Stack-Edu-Python**: Educational Python code samples

Despite their small size, SmolLM models demonstrate strong performance across reasoning and coding tasks, making them ideal for applications where efficiency and performance need to be balanced.

## 🛠️ Technical Implementation Details

### Session State Management

The application uses Streamlit's session state to maintain code between interactions:
- Session variables for generated code, output, and errors
- Editor key rotation to force UI refreshes when code changes
- State tracking for execution results

### Code Execution Security

Code execution is handled securely through:
- Isolated environment with limited global variables
- Execution in controlled context with output/error redirection
- Exception handling to prevent application crashes

### Device Optimization

The model loader automatically selects the optimal device:
- CUDA for NVIDIA GPUs
- MPS for Apple Silicon
- CPU fallback for other environments

### Logging System

Comprehensive logging throughout the application:
- Configurable log levels
- Detailed event tracking for debugging
- Error capture with context information

## ⚙️ Fine-Tuning Process

The SmolLM model used in this application was fine-tuned using:

1. **Supervised Fine-Tuning (SFT)**: Training on pairs of docstrings and corresponding Python code
2. **Parameter-Efficient Fine-Tuning**: Using LoRA to efficiently adapt model weights
3. **Checkpoint Selection**: Evaluating model performance across training checkpoints

This fine-tuning approach allows the model to generate high-quality Python code while maintaining the small model size that enables local execution.

## ⚠️ Limitations and Considerations

- The model is limited to generating relatively simple Python code snippets
- Code execution happens in the Streamlit environment - use caution with untrusted code
- Large code generations may require adjusting the max tokens parameter
- The model performs best on standard programming tasks and may struggle with domain-specific libraries

## 🔍 Troubleshooting

- **Model Loading Issues**: Ensure the model checkpoint directory exists and has proper permissions
- **CUDA/MPS Errors**: Check your PyTorch installation supports your hardware
- **Package Installation Failures**: Verify you have proper permissions and connectivity
- **Code Execution Errors**: Look for missing imports or environment-specific issues

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 🙏 Acknowledgments

- HuggingFace for the SmolLM base model
- Streamlit team for the amazing framework
- CodeSearchNet for the training data
- [Astral](https://astral.sh/) for the uv package manager
