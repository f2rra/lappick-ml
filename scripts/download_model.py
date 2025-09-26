from huggingface_hub import hf_hub_download
import os

# Define the directory to save the model
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "..", "models")

# Ensure the model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# Download the model file from Hugging Face Hub
model_path = hf_hub_download(
    repo_id="f2rra/universal-sentence-encoder",
    filename="variables.data-00000-of-00001",
    repo_type="model",
    local_dir=MODEL_DIR
)
print(f"Model downloaded to: {model_path}")
