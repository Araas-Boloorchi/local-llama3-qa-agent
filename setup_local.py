import os
from huggingface_hub import hf_hub_download

def download_model():
    model_name = "bartowski/Llama-3.2-3B-Instruct-GGUF"
    filename = "Llama-3.2-3B-Instruct-Q4_K_M.gguf"
    
    print(f"Downloading {filename} from {model_name}...")
    
    # Create models directory if it doesn't exist
    if not os.path.exists("models"):
        os.makedirs("models")
        
    try:
        model_path = hf_hub_download(
            repo_id=model_name,
            filename=filename,
            local_dir="models",
            local_dir_use_symlinks=False
        )
        print(f"Successfully downloaded model to: {model_path}")
        return model_path
    except Exception as e:
        print(f"Error downloading model: {e}")
        return None

if __name__ == "__main__":
    download_model()
