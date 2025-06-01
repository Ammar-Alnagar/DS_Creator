"""Hugging Face Hub integration for uploading datasets."""

import json
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

from loguru import logger
from huggingface_hub import HfApi, Repository, login, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

from config import config


class HuggingFaceUploader:
    """Handles uploading datasets to Hugging Face Hub."""
    
    def __init__(self):
        self.api = None
        self._authenticated = False
    
    def authenticate(self) -> bool:
        """
        Authenticate with Hugging Face Hub using the API key from config.
        
        Returns:
            bool: True if authentication successful, False otherwise
        """
        if not config.huggingface_api_key:
            logger.error("Hugging Face API key not found. Please set HUGGINGFACE_API_KEY in your .env file")
            return False
        
        try:
            # Login using the API key
            login(token=config.huggingface_api_key, add_to_git_credential=True)
            self.api = HfApi()
            
            # Test authentication by getting user info
            user_info = self.api.whoami()
            logger.info(f"Successfully authenticated with Hugging Face as: {user_info['name']}")
            self._authenticated = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to authenticate with Hugging Face: {e}")
            return False
    
    def create_dataset_repo(self, repo_name: str, private: bool = False) -> bool:
        """
        Create a new dataset repository on Hugging Face Hub.
        
        Args:
            repo_name: Name of the repository (format: username/dataset-name)
            private: Whether the repository should be private
            
        Returns:
            bool: True if repository created or already exists, False on error
        """
        if not self._authenticated:
            logger.error("Not authenticated with Hugging Face. Call authenticate() first.")
            return False
        
        try:
            # Check if repo already exists
            try:
                repo_info = self.api.dataset_info(repo_name)
                logger.info(f"Dataset repository '{repo_name}' already exists")
                return True
            except RepositoryNotFoundError:
                # Repository doesn't exist, create it
                pass
            
            # Create the repository
            create_repo(
                repo_id=repo_name,
                repo_type="dataset",
                private=private,
                exist_ok=True
            )
            logger.info(f"Created dataset repository: {repo_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create repository '{repo_name}': {e}")
            return False
    
    def upload_dataset(
        self, 
        dataset_path: Path, 
        repo_name: Optional[str] = None,
        private: Optional[bool] = None,
        commit_message: Optional[str] = None
    ) -> bool:
        """
        Upload a dataset file to Hugging Face Hub.
        
        Args:
            dataset_path: Path to the dataset JSON file
            repo_name: Repository name (defaults to config.hf_repo_name)
            private: Whether repository should be private (defaults to config.hf_dataset_private)
            commit_message: Commit message (defaults to config.hf_commit_message)
            
        Returns:
            bool: True if upload successful, False otherwise
        """
        if not self._authenticated:
            if not self.authenticate():
                return False
        
        # Use config defaults if not provided
        repo_name = repo_name or config.hf_repo_name
        private = private if private is not None else config.hf_dataset_private
        commit_message = commit_message or config.hf_commit_message
        
        if not repo_name:
            logger.error("Repository name not specified. Set HF_REPO_NAME in .env or provide repo_name parameter")
            return False
        
        if not dataset_path.exists():
            logger.error(f"Dataset file not found: {dataset_path}")
            return False
        
        try:
            # Create repository if it doesn't exist
            if not self.create_dataset_repo(repo_name, private):
                return False
            
            # Prepare files to upload
            files_to_upload = [dataset_path]
            
            # Also upload metadata file if it exists
            metadata_path = dataset_path.with_suffix('.metadata.json')
            if metadata_path.exists():
                files_to_upload.append(metadata_path)
            
            # Create README.md with dataset information
            readme_content = self._generate_readme(dataset_path, metadata_path)
            readme_path = dataset_path.parent / "README.md"
            with open(readme_path, 'w', encoding='utf-8') as f:
                f.write(readme_content)
            files_to_upload.append(readme_path)
            
            # Upload files
            logger.info(f"Uploading {len(files_to_upload)} files to {repo_name}...")
            
            for file_path in files_to_upload:
                logger.info(f"Uploading {file_path.name}...")
                self.api.upload_file(
                    path_or_fileobj=str(file_path),
                    path_in_repo=file_path.name,
                    repo_id=repo_name,
                    repo_type="dataset",
                    commit_message=f"{commit_message} - {file_path.name}"
                )
            
            # Clean up temporary README
            if readme_path.exists():
                readme_path.unlink()
            
            logger.info(f"Successfully uploaded dataset to: https://huggingface.co/datasets/{repo_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to upload dataset: {e}")
            return False
    
    def _generate_readme(self, dataset_path: Path, metadata_path: Optional[Path] = None) -> str:
        """Generate a README.md file for the dataset."""
        
        # Load basic dataset info
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                dataset = json.load(f)
            
            num_records = len(dataset)
            
            # Try to determine format
            if dataset and isinstance(dataset[0], dict):
                if 'conversations' in dataset[0]:
                    format_type = "ChatML (Conversations)"
                    structure_desc = "Each record contains an ID and a conversations array with human/assistant message pairs."
                elif 'instruction' in dataset[0]:
                    format_type = "Instruction Tuning"
                    structure_desc = "Each record contains instruction, input, and output fields for instruction tuning."
                else:
                    format_type = "Custom"
                    structure_desc = "Custom format dataset."
            else:
                format_type = "Unknown"
                structure_desc = "Unknown format."
                
        except Exception:
            num_records = "Unknown"
            format_type = "Unknown"
            structure_desc = "Could not determine dataset structure."
        
        # Load metadata if available
        stats_info = ""
        if metadata_path and metadata_path.exists():
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                if 'generation_stats' in metadata:
                    stats = metadata['generation_stats']
                    stats_info = f"""
## Generation Statistics

- **PDFs Processed:** {stats.get('total_pdfs_processed', 'N/A')}
- **Text Chunks Extracted:** {stats.get('total_chunks_extracted', 'N/A')}
- **Conversations Generated:** {stats.get('total_conversations_generated', 'N/A')}
- **Success Rate:** {((stats.get('successful_generations', 0) / max(stats.get('total_chunks_extracted', 1), 1)) * 100):.1f}%
- **Average Confidence Score:** {stats.get('average_confidence_score', 'N/A'):.2f}
- **Processing Time:** {stats.get('processing_time_seconds', 'N/A'):.1f} seconds
"""
            except Exception:
                pass
        
        readme_content = f"""---
license: apache-2.0
task_categories:
- conversational
- question-answering
language:
- en
tags:
- medical
- healthcare
- conversations
- synthetic
size_categories:
- n<1K
---

# Medical Conversation Dataset

This dataset contains synthetic medical conversations generated from medical literature and documents.

## Dataset Information

- **Format:** {format_type}
- **Number of Records:** {num_records}
- **Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")}

## Structure

{structure_desc}

{stats_info}

## Usage

This dataset is designed for training conversational AI models for medical applications. It should be used responsibly and always in conjunction with proper medical disclaimers.

### Loading the Dataset

```python
import json

# Load the dataset
with open('dataset_file.json', 'r') as f:
    dataset = json.load(f)

# Access conversations
for record in dataset:
    # Process based on format
    pass
```

## Important Medical Disclaimer

⚠️ **This dataset is for educational and research purposes only. The generated conversations should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare professionals for medical concerns.**

## License

Apache 2.0

## Citation

If you use this dataset, please cite:

```
@dataset{{medical_conversations_{datetime.now().strftime("%Y")},
  title={{Medical Conversation Dataset}},
  author={{Generated using DS_Creator}},
  year={{{datetime.now().year}}},
  url={{https://huggingface.co/datasets/[repo_name]}}
}}
```
"""
        
        return readme_content 