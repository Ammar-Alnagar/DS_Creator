"""Configuration management for the Medical Dataset Creator."""

import os
from pathlib import Path
from typing import Optional
from pydantic import Field, validator
from pydantic_settings import BaseSettings


class Config(BaseSettings):
    """Application configuration using Pydantic for validation."""
    
    # API Keys - Updated for new providers
    deepseek_api_key: Optional[str] = Field(None, env="DEEPSEEK_API_KEY")
    openai_api_key: Optional[str] = Field(None, env="OPENAI_API_KEY")  # For OpenAI compatible APIs
    huggingface_api_key: Optional[str] = Field(None, env="HUGGINGFACE_API_KEY")
    
    # OpenAI-compatible API settings (for OpenRouter, etc.)
    openai_base_url: Optional[str] = Field(None, env="OPENAI_BASE_URL")  # e.g., "https://openrouter.ai/api/v1"
    openai_organization: Optional[str] = Field(None, env="OPENAI_ORGANIZATION")
    
    # Legacy support (deprecated but kept for backward compatibility)
    gemini_api_key: Optional[str] = Field(None, env="GEMINI_API_KEY")
    
    # Model Configuration
    api_provider: str = Field("deepseek", env="API_PROVIDER")  # deepseek, openai, or local
    default_model: str = Field("deepseek-chat", env="DEFAULT_MODEL")
    local_model_name: str = Field("microsoft/DialoGPT-medium", env="LOCAL_MODEL_NAME")
    use_local_model: bool = Field(False, env="USE_LOCAL_MODEL")
    temperature: float = Field(0.7, env="TEMPERATURE")
    max_tokens: int = Field(1000, env="MAX_TOKENS")
    
    # Dataset Format Configuration
    dataset_format: str = Field("chatml", env="DATASET_FORMAT")  # chatml or instruction
    
    # Hugging Face Upload Configuration
    enable_hf_upload: bool = Field(False, env="ENABLE_HF_UPLOAD")
    hf_repo_name: Optional[str] = Field(None, env="HF_REPO_NAME")  # e.g., "username/dataset-name"
    hf_dataset_private: bool = Field(False, env="HF_DATASET_PRIVATE")
    hf_commit_message: str = Field("Upload medical conversation dataset", env="HF_COMMIT_MESSAGE")
    
    # Generation Settings
    max_conversations_per_chunk: int = Field(5, env="MAX_CONVERSATIONS_PER_CHUNK")
    min_conversation_length: int = Field(50, env="MIN_CONVERSATION_LENGTH")
    max_conversation_length: int = Field(500, env="MAX_CONVERSATION_LENGTH")
    
    # Processing Settings
    chunk_size: int = Field(2000, env="CHUNK_SIZE")
    chunk_overlap: int = Field(200, env="CHUNK_OVERLAP")
    batch_size: int = Field(10, env="BATCH_SIZE")
    
    # Local Model Settings
    device: str = Field("auto", env="DEVICE")  # auto, cpu, cuda
    max_memory_gb: int = Field(8, env="MAX_MEMORY_GB")
    load_in_8bit: bool = Field(True, env="LOAD_IN_8BIT")
    
    # Paths
    output_dir: Path = Field(Path("./datasets"), env="OUTPUT_DIR")
    input_dir: Path = Field(Path("./input_pdfs"), env="INPUT_DIR")
    models_cache_dir: Path = Field(Path("./models_cache"), env="MODELS_CACHE_DIR")
    
    # Logging
    log_level: str = Field("INFO", env="LOG_LEVEL")
    
    @validator('api_provider', 'default_model', 'local_model_name', 'device', 'log_level', 'enable_hf_upload', 'hf_dataset_private', 'hf_repo_name', pre=True)
    def strip_comments(cls, v):
        """Strip inline comments from string configuration values."""
        if isinstance(v, str):
            # Split on '#' and take the first part, then strip whitespace
            cleaned = v.split('#')[0].strip()
            # For boolean fields, convert common string representations
            if cleaned.lower() in ('true', '1', 'yes', 'on'):
                return True
            elif cleaned.lower() in ('false', '0', 'no', 'off'):
                return False
            return cleaned
        return v
    
    @validator('openai_base_url', pre=True)
    def strip_comments_from_url(cls, v):
        """Strip inline comments from URL configuration values."""
        if isinstance(v, str):
            return v.split('#')[0].strip()
        return v
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global configuration instance
config = Config()

# Create necessary directories
config.output_dir.mkdir(exist_ok=True)
config.input_dir.mkdir(exist_ok=True)
config.models_cache_dir.mkdir(exist_ok=True) 