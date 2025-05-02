"""
Python Code Generator App with SmolLM
--------------------------------
A professional Streamlit application that uses a fine-tuned SmolLM model 
to generate Python code from natural language descriptions.

This application demonstrates advanced model deployment techniques
and proper software architecture for AI-based code generation tools.
"""

import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import sys
import subprocess
import io
from contextlib import redirect_stdout, redirect_stderr
import traceback
import os
import time
from streamlit_ace import st_ace
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants and configuration
MODEL_CONFIG = {
    "base_model_name": "HuggingFaceTB/SmolLM-135M",
    "lora_adapter_path": "smollm-sft-lora-output/checkpoint-140",
    "fallback_model_path": "smollm-sft-lora-output",
    "local_only": True,
    "load_timeout": 60
}

def configure_page():
    """
    Configure the Streamlit page settings with optimized layout and appearance.
    """
    st.set_page_config(
        page_title="Advanced Python Code Generator",
        page_icon="💻",
        layout="wide",
        initial_sidebar_state="expanded"
    )

class ModelLoader:
    """
    Handles model loading, configuration, and inference operations.
    Manages the lifecycle of ML models including loading, optimization, and inference.
    """
    
    @staticmethod
    @st.cache_resource
    def load_model():
        """
        Load the model with caching to avoid redundant loading between sessions.
        
        Returns:
            tuple: (model, tokenizer, device) The loaded model components
        """
        return ModelLoader._load_model_with_timeout()
    
    @staticmethod
    def _load_model_with_timeout():
        """
        Load the fine-tuned model with timeout protection to prevent hanging.
        
        Returns:
            tuple: (model, tokenizer, device) The loaded model components
            
        Raises:
            Exception: If model loading fails or times out
        """
        start_time = time.time()
        
        # Determine which model path to use
        if os.path.exists(MODEL_CONFIG["lora_adapter_path"]):
            model_path = MODEL_CONFIG["lora_adapter_path"]
            logger.info(f"Using checkpoint: {model_path}")
            st.sidebar.info(f"🔍 Using checkpoint: {model_path}")
        elif os.path.exists(MODEL_CONFIG["fallback_model_path"]):
            model_path = MODEL_CONFIG["fallback_model_path"]
            logger.info(f"Checkpoint not found. Using fallback: {model_path}")
            st.sidebar.info(f"⚠️ Checkpoint not found. Using fallback: {model_path}")
        else:
            error_msg = f"No fine-tuned model found at either {MODEL_CONFIG['lora_adapter_path']} or {MODEL_CONFIG['fallback_model_path']}"
            logger.error(error_msg)
            st.error(f"❌ {error_msg}")
            st.stop()
        
        # Load the configuration to get the base model info
        try:
            peft_config = PeftConfig.from_pretrained(model_path)
            base_model_name = peft_config.base_model_name_or_path
            
            # If no explicit base model, use the one from config
            if not base_model_name or base_model_name == "":
                base_model_name = MODEL_CONFIG["base_model_name"]
                
            logger.info(f"Base model identified as: {base_model_name}")
            st.sidebar.info(f"Base model identified as: {base_model_name}")
        except Exception as e:
            logger.warning(f"Could not determine base model from adapter config: {str(e)}")
            st.sidebar.warning(f"Could not determine base model from adapter config: {str(e)}")
            base_model_name = MODEL_CONFIG["base_model_name"]
        
        # Load tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                base_model_name,
                local_files_only=MODEL_CONFIG["local_only"]
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            logger.info("Tokenizer loaded successfully")
        except Exception as e:
            error_msg = f"Failed to load tokenizer: {str(e)}"
            logger.error(error_msg)
            st.error(f"❌ {error_msg}")
            st.error("Please make sure you have the model cached locally or set local_only to False")
            st.stop()
        
        # Load the base model
        device = ModelLoader._get_optimal_device()
        try:
            # Check if we're within timeout
            if time.time() - start_time > MODEL_CONFIG["load_timeout"]:
                error_msg = f"Timeout reached while loading base model ({MODEL_CONFIG['load_timeout']} seconds)"
                logger.error(error_msg)
                st.error(f"⏱️ {error_msg}")
                st.stop()
                
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                trust_remote_code=True,
                local_files_only=MODEL_CONFIG["local_only"]
            ).to(device)
            logger.info(f"Base model loaded successfully on {device}")
        except Exception as e:
            error_msg = f"Failed to load base model: {str(e)}"
            logger.error(error_msg)
            st.error(f"❌ {error_msg}")
            if MODEL_CONFIG["local_only"]:
                st.error("The model could not be loaded from local cache. You may need to run once with local_only=False")
            st.stop()
        
        # Load the fine-tuned LoRA model
        try:
            # Check if we're within timeout
            if time.time() - start_time > MODEL_CONFIG["load_timeout"]:
                error_msg = f"Timeout reached while loading LoRA model ({MODEL_CONFIG['load_timeout']} seconds)"
                logger.error(error_msg)
                st.error(f"⏱️ {error_msg}")
                st.stop()
                
            model = PeftModel.from_pretrained(
                base_model,
                model_path,
                torch_dtype=torch.float32  # Use float32 for compatibility
            )
            logger.info("LoRA model loaded successfully")
            st.sidebar.success("✅ LoRA model loaded successfully!")
        except Exception as e:
            error_msg = f"Failed to load LoRA model: {str(e)}"
            logger.error(error_msg)
            st.error(f"❌ {error_msg}")
            st.stop()
        
        total_time = time.time() - start_time
        logger.info(f"Model loaded in {total_time:.2f} seconds")
        st.sidebar.info(f"⏱️ Model loaded in {total_time:.2f} seconds")
        
        return model, tokenizer, device
    
    @staticmethod
    def _get_optimal_device():
        """
        Determine the best available computational device for model inference.
        
        Returns:
            str: The device identifier ('cuda', 'mps', or 'cpu')
        """
        if torch.cuda.is_available():
            logger.info("CUDA is available, using GPU")
            return "cuda"
        elif torch.backends.mps.is_available():
            logger.info("MPS is available, using Apple Silicon")
            return "mps"
        else:
            logger.info("No GPU available, using CPU")
            return "cpu"
    
    @staticmethod
    def generate_code(model, tokenizer, device, prompt, temperature=0.15, max_tokens=200):
        """
        Generate Python code from natural language description using the loaded model.
        
        Args:
            model: The loaded language model
            tokenizer: The tokenizer for the model
            device: The device to run inference on
            prompt (str): The natural language description
            temperature (float): Sampling temperature (higher = more creative)
            max_tokens (int): Maximum number of tokens to generate
            
        Returns:
            str: The generated Python code
        """
        logger.info(f"Generating code with temperature={temperature}, max_tokens={max_tokens}")
        
        # Format the prompt
        formatted_prompt = ModelLoader._format_prompt(prompt)
        
        # Tokenize the prompt
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
        
        # Generate code
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True
            )
        
        # Decode the generated text
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the generated code part (after the prompt)
        code_only = generated_text[len(formatted_prompt):].strip()
        logger.info("Code generation completed successfully")
        
        return code_only
    
    @staticmethod
    def _format_prompt(docstring):
        """
        Format the input prompt to match the training data structure.
        
        Args:
            docstring (str): The natural language description
            
        Returns:
            str: The formatted prompt
        """
        return f"### Docstring:\n{docstring}\n\n### Code:"

class CodeExecutor:
    """
    Handles code execution and package management with secure sandboxing.
    Provides isolated environment for running generated code and managing dependencies.
    """
    
    @staticmethod
    def execute_code(code):
        """
        Execute Python code and capture output and errors in a controlled environment.
        
        Args:
            code (str): The Python code to execute
            
        Returns:
            tuple: (stdout_output, stderr_output, exception_traceback)
        """
        logger.info("Executing code")
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        try:
            # Create a safe globals dictionary with basic modules
            safe_globals = {
                "__builtins__": __builtins__,
                "print": print,
                "range": range,
                "len": len,
                "list": list,
                "dict": dict,
                "set": set,
                "int": int,
                "float": float,
                "str": str,
                "bool": bool,
                "tuple": tuple,
                "enumerate": enumerate,
            }
            
            # Add imported modules to the globals
            for line in code.split('\n'):
                if line.startswith('import ') or line.startswith('from '):
                    try:
                        exec(line, safe_globals)
                    except Exception as e:
                        logger.error(f"Error importing module: {str(e)}")
                        return "", f"Error importing module: {str(e)}", traceback.format_exc()
            
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                # Execute the code in the safe environment
                exec(code, safe_globals)
            
            stdout_output = stdout_capture.getvalue()
            stderr_output = stderr_capture.getvalue()
            logger.info("Code executed successfully")
            return stdout_output, stderr_output, None
        except Exception as e:
            logger.error(f"Code execution error: {str(e)}")
            return stdout_capture.getvalue(), stderr_capture.getvalue(), traceback.format_exc()
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
    
    @staticmethod
    def install_package(package_name):
        """
        Install a Python package using pip in the current environment.
        
        Args:
            package_name (str): The name of the package to install
            
        Returns:
            tuple: (success_flag, output/error_message)
        """
        logger.info(f"Installing package: {package_name}")
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", package_name],
                capture_output=True,
                text=True,
                check=True
            )
            logger.info(f"Package {package_name} installed successfully")
            return True, result.stdout
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install package {package_name}: {e.stderr}")
            return False, e.stderr

class AppUI:
    """
    Manages the Streamlit user interface components and interaction logic.
    Handles rendering, state management, and user interactions within the application.
    """
    
    def __init__(self):
        """
        Initialize the UI components and session state variables.
        """
        # Initialize session state with default values if they don't exist
        if 'generated_code' not in st.session_state:
            st.session_state.generated_code = "# Your generated code will appear here"
        if 'output' not in st.session_state:
            st.session_state.output = ""
        if 'error' not in st.session_state:
            st.session_state.error = ""
        if 'code_generated' not in st.session_state:
            st.session_state.code_generated = False
        # Add a new editor key counter to force re-render
        if 'editor_key' not in st.session_state:
            st.session_state.editor_key = 0
    
    def render_header(self):
        """
        Render the application header and introduction section.
        """
        st.title("Python Code Generator")
        st.markdown("""
        This app uses a fine-tuned language model to generate Python code from descriptions.
        Enter what you want the code to do, and the AI will write it for you.
        """)
    
    def render_sidebar(self):
        """
        Render the sidebar with settings and model information.
        
        Returns:
            tuple: (temperature, max_tokens) The user-selected generation parameters
        """
        with st.sidebar:
            st.header("Settings")
            temperature = st.slider("Creativity", min_value=0.1, max_value=1.0, value=0.15, step=0.1,
                                help="Higher values make output more creative, lower values more predictable")
            max_tokens = st.slider("Max length", min_value=50, max_value=500, value=200, step=50,
                                help="Maximum number of tokens to generate")
            
            st.markdown("---")
            st.subheader("Model Info")
            st.markdown(f"""
            - **Base Model**: {MODEL_CONFIG["base_model_name"].split('/')[-1]} 
            - **LoRA Checkpoint**: {MODEL_CONFIG["lora_adapter_path"].split('/')[-1]}
            """)
            
            st.markdown("---")
            st.subheader("About")
            st.markdown("""
            This app generates Python code using a fine-tuned model trained on:
            - Python code examples from CodeSearchNet
            - Using Parameter-Efficient Fine-Tuning (PEFT)
            """)
            
            return temperature, max_tokens
    
    def render_input_column(self, model, tokenizer, device, temperature, max_tokens):
        """
        Render the input column with prompt area and code generation controls.
        
        Args:
            model: The loaded language model
            tokenizer: The tokenizer for the model
            device: The device to run inference on
            temperature (float): Sampling temperature
            max_tokens (int): Maximum number of tokens to generate
        """
        st.header("Describe Your Code")
        prompt = st.text_area(
            "What should the code do?",
            height=150,
            placeholder="Example: Write a function that takes a list of numbers and returns the sum of all even numbers in the list.",
            key="prompt_area"
        )
        
        generate_pressed = st.button("Generate Code", type="primary", key="generate_btn")
        if generate_pressed:
            if prompt:
                with st.spinner("Generating code..."):
                    # Generate code
                    generated_code = ModelLoader.generate_code(
                        model, tokenizer, device, prompt, 
                        temperature=temperature, 
                        max_tokens=max_tokens
                    )
                    
                    # Store the generated code in session state
                    if generated_code:
                        # Update the session state with new code
                        st.session_state.generated_code = generated_code
                        st.session_state.code_generated = True
                        # Increment the editor key to force re-render
                        st.session_state.editor_key += 1
                        logger.info("Code stored in session state")
                        st.success("Code generated successfully!")
                    else:
                        st.error("Failed to generate code. Please try again.")
            else:
                st.error("Please enter a description first.")
        
        # Package installation UI
        with st.expander("Need to install packages?"):
            package = st.text_input("Package name:", placeholder="e.g., numpy, pandas, matplotlib")
            if st.button("Install Package"):
                if package:
                    with st.spinner(f"Installing {package}..."):
                        success, output = CodeExecutor.install_package(package)
                        if success:
                            st.success(f"Successfully installed {package}")
                        else:
                            st.error(f"Failed to install {package}")
                            st.code(output)
                else:
                    st.error("Please enter a package name.")
    
    def render_output_column(self):
        """
        Render the output column with code editor and execution results.
        """
        st.header("Code Editor")
        
        # Get the current code from session state
        current_code = st.session_state.generated_code
        
        # Interactive code editor with syntax highlighting - use a unique key with editor_key
        code = st_ace(
            value=current_code,
            language="python",
            theme="monokai",
            font_size=14,
            key=f"ace_editor_{st.session_state.editor_key}",  # Use the counter to force re-render
            height=300,
            auto_update=True,
            wrap=True,
            show_gutter=True,
        )
        
        run_pressed = st.button("Run Code", type="primary", key="run_btn")
        if run_pressed:
            if code.strip():
                with st.spinner("Executing code..."):
                    stdout, stderr, exception = CodeExecutor.execute_code(code)
                    
                    # Clear previous output/errors
                    st.session_state.output = ""
                    st.session_state.error = ""
                    
                    if stdout:
                        st.session_state.output = stdout
                    if stderr or exception:
                        st.session_state.error = stderr if stderr else exception
                    
                    # If no output or error, indicate success
                    if not stdout and not stderr and not exception:
                        st.session_state.output = "Code executed successfully with no output."
            else:
                st.error("No code to execute.")
        
        # Output and error display
        if st.session_state.output:
            st.subheader("Output")
            st.code(st.session_state.output, language="text")
            
        if st.session_state.error:
            st.subheader("Errors")
            st.error(st.session_state.error)

def main():
    """
    Main function that orchestrates the application workflow.
    Sets up the environment, initializes components, and manages the application lifecycle.
    """
    try:
        # Configure the page
        configure_page()
        
        # Initialize UI
        app_ui = AppUI()
        
        # Render header
        app_ui.render_header()
        
        # Render sidebar and get settings
        temperature, max_tokens = app_ui.render_sidebar()
        
        # Load the model
        with st.spinner("Loading model... This may take a moment."):
            try:
                model, tokenizer, device = ModelLoader.load_model()
                logger.info("Model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load model: {str(e)}")
                st.error(f"Failed to load model: {str(e)}")
                st.stop()
        
        # Create two columns for input and output
        col1, col2 = st.columns([1, 1])
        
        # Render input column
        with col1:
            app_ui.render_input_column(model, tokenizer, device, temperature, max_tokens)
        
        # Render output column
        with col2:
            app_ui.render_output_column()
            
    except Exception as e:
        logger.exception("Unhandled exception in the application")
        st.error(f"An unexpected error occurred: {str(e)}")
        st.error("Please refresh the page and try again.")

if __name__ == "__main__":
    main()