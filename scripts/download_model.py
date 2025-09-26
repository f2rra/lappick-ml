from huggingface_hub import snapshot_download
import os

# Define the directory to save the model
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "..", "models")

# Ensure the model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# Download the model file from Hugging Face Hub
model_path = snapshot_download(
    repo_id="f2rra/universal-sentence-encoder",
    local_dir=MODEL_DIR,
    repo_type="model"
)
print(f"Model downloaded to: {model_path}")
