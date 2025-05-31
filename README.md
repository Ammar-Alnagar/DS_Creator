# 🏥 Medical Dataset Creator

**Generate high-quality medical conversation datasets from PDF documents using AI models**

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![DeepSeek](https://img.shields.io/badge/AI-DeepSeek-purple.svg)](https://www.deepseek.com/)
[![OpenAI](https://img.shields.io/badge/AI-OpenAI%20Compatible-cyan.svg)](https://openrouter.ai/)

## 🚀 What's New in v2.0

- **🔥 DeepSeek Integration**: High-quality, cost-effective AI conversations
- **🌐 OpenAI-Compatible APIs**: Support for OpenRouter, OpenAI, and other providers
- **⚡ Improved Performance**: Faster generation with better quality control
- **🔧 Enhanced Configuration**: More flexible model selection and settings

---

## 📋 Table of Contents

- [🎯 Features](#-features)
- [🏃 Quick Start](#-quick-start)
- [⚙️ Configuration](#️-configuration)
- [🤖 AI Models](#-ai-models)
- [💻 Usage Examples](#-usage-examples)
- [📊 Output Format](#-output-format)
- [🔧 Advanced Usage](#-advanced-usage)
- [🐛 Troubleshooting](#-troubleshooting)
- [🤝 Contributing](#-contributing)

---

## 🎯 Features

### 🧠 **Multi-Provider AI Support**
- **DeepSeek API**: Cost-effective, high-quality conversations
- **OpenAI-Compatible APIs**: OpenRouter, OpenAI, Claude via OpenRouter
- **Local Models**: Privacy-focused Hugging Face models

### 📚 **Advanced PDF Processing**
- Multiple extraction methods (PyPDF2, pdfplumber, PyMuPDF)
- Intelligent text chunking with overlap
- Quality scoring and filtering

### 🎛️ **Flexible Generation**
- Configurable conversation length and style
- Batch processing with progress tracking
- Medical disclaimers and safety guidelines

### 📈 **Quality Assurance**
- Automated quality scoring
- Content filtering and validation
- Comprehensive metadata tracking

---

## 🏃 Quick Start

### 1️⃣ **Installation**

```bash
# Clone the repository
git clone https://github.com/Ammar-Alnagar/Dataset-Creator
cd Dataset-Creator

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2️⃣ **Configuration**

Create a `.env` file with your API credentials:

```bash
# Copy sample configuration
cp env.sample .env

# Edit with your API keys
nano .env
```

**DeepSeek Setup (Recommended):**
```env
API_PROVIDER=deepseek
DEEPSEEK_API_KEY=your_deepseek_api_key_here
DEFAULT_MODEL=deepseek-chat
```

**OpenRouter Setup:**
```env
API_PROVIDER=openai
OPENAI_API_KEY=your_openrouter_api_key_here
OPENAI_BASE_URL=https://openrouter.ai/api/v1
DEFAULT_MODEL=anthropic/claude-3-sonnet
```

**OpenAI Setup:**
```env
API_PROVIDER=openai
OPENAI_API_KEY=your_openai_api_key_here
DEFAULT_MODEL=gpt-4
```

### 3️⃣ **Generate Your First Dataset**

```bash
# Quick sample (50 conversations)
python main.py generate --sample

# Full generation
python main.py generate --max-conversations 1000

# Check system status
python main.py config-check
```

---

## ⚙️ Configuration

### 🔑 **API Provider Options**

| Provider | Cost | Quality | Privacy | Setup Difficulty |
|----------|------|---------|---------|------------------|
| **DeepSeek** | 💰 Low | ⭐⭐⭐⭐⭐ | ☁️ Cloud | 🟢 Easy |
| **OpenRouter** | 💰💰 Medium | ⭐⭐⭐⭐⭐ | ☁️ Cloud | 🟢 Easy |
| **OpenAI** | 💰💰💰 High | ⭐⭐⭐⭐⭐ | ☁️ Cloud | 🟢 Easy |
| **Local Models** | 🆓 Free | ⭐⭐⭐ | 🔒 Private | 🟡 Medium |

### 📋 **Environment Variables**

```env
# Primary Configuration
API_PROVIDER=deepseek                    # deepseek, openai, or local
DEFAULT_MODEL=deepseek-chat              # Model name
TEMPERATURE=0.7                          # Creativity (0.0-1.0)
MAX_TOKENS=1000                          # Response length

# DeepSeek
DEEPSEEK_API_KEY=your_key_here

# OpenAI-Compatible
OPENAI_API_KEY=your_key_here
OPENAI_BASE_URL=https://openrouter.ai/api/v1  # Optional
OPENAI_ORGANIZATION=your_org_id               # Optional

# Generation Settings
MAX_CONVERSATIONS_PER_CHUNK=5            # Conversations per text chunk
MIN_CONVERSATION_LENGTH=50               # Minimum message length
MAX_CONVERSATION_LENGTH=500              # Maximum message length

# Processing
CHUNK_SIZE=2000                          # Text chunk size
CHUNK_OVERLAP=200                        # Overlap between chunks
BATCH_SIZE=10                            # Processing batch size

# Local Models (for privacy)
USE_LOCAL_MODEL=false
LOCAL_MODEL_NAME=microsoft/DialoGPT-medium
DEVICE=auto                              # auto, cpu, cuda, mps
LOAD_IN_8BIT=true                        # Memory optimization
```

---

## 🤖 AI Models

### ☁️ **Cloud Models (Recommended)**

#### **DeepSeek Models**
```env
DEFAULT_MODEL=deepseek-chat              # General conversations
DEFAULT_MODEL=deepseek-coder             # Technical content
```

#### **OpenRouter Models**
```env
DEFAULT_MODEL=anthropic/claude-3-sonnet  # Excellent reasoning
DEFAULT_MODEL=google/gemini-pro-1.5      # Long context
DEFAULT_MODEL=openai/gpt-4               # High quality
DEFAULT_MODEL=meta-llama/llama-3-70b     # Open source
```

#### **OpenAI Models**
```env
DEFAULT_MODEL=gpt-4                      # Highest quality
DEFAULT_MODEL=gpt-3.5-turbo             # Fast and affordable
```

### 🏠 **Local Models (Privacy-Focused)**

```env
USE_LOCAL_MODEL=true
LOCAL_MODEL_NAME=microsoft/DialoGPT-medium     # ~1.5GB, good quality
LOCAL_MODEL_NAME=microsoft/DialoGPT-large      # ~3GB, better quality
LOCAL_MODEL_NAME=facebook/blenderbot-400M-distill  # ~1.6GB, optimized
```

---

## 💻 Usage Examples

### 🎯 **Basic Generation**

```bash
# Generate sample dataset
python main.py generate --sample

# Generate from specific directory
python main.py generate --input-dir ./medical_books --max-conversations 500

# Use custom output filename
python main.py generate --output-file cardiology_dataset.json --max-conversations 1000
```

### 🔍 **Analysis & Validation**

```bash
# Analyze PDFs without generation
python main.py analyze --input-dir ./medical_pdfs

# Validate existing dataset
python main.py validate ./datasets/my_dataset.json

# Check system configuration
python main.py config-check
```

### 🏠 **Local Model Management**

```bash
# List available models
python main.py local-models --list-models

# Test specific model
python main.py local-models microsoft/DialoGPT-medium

# Force local model usage
python main.py generate --use-local --max-conversations 100
```

### ⚙️ **Advanced Options**

```bash
# High-quality generation with specific settings
API_PROVIDER=deepseek \
TEMPERATURE=0.5 \
MAX_CONVERSATIONS_PER_CHUNK=3 \
python main.py generate --max-conversations 2000

# Privacy-focused generation
USE_LOCAL_MODEL=true \
LOCAL_MODEL_NAME=microsoft/DialoGPT-large \
python main.py generate --use-local --max-conversations 500
```

---

## 📊 Output Format

### 📋 **Dataset Structure**

```json
{
  "conversations": [
    [
      {
        "from": "user",
        "value": "I've been having chest pain during exercise. Should I be concerned?"
      },
      {
        "from": "assistant", 
        "value": "Chest pain during exercise can be a serious symptom that may indicate heart problems. I strongly recommend consulting with a cardiologist immediately for proper evaluation. They may perform tests like an ECG or stress test to determine the cause. Please don't ignore this symptom and seek medical attention promptly."
      }
    ]
  ]
}
```

### 📊 **Metadata File**

```json
{
  "generation_stats": {
    "total_pdfs_processed": 15,
    "total_conversations_generated": 1250,
    "average_confidence_score": 0.87,
    "processing_time_seconds": 1847.3,
    "models_used": ["deepseek:deepseek-chat"]
  },
  "configuration": {
    "chunk_size": 2000,
    "temperature": 0.7,
    "max_tokens": 1000
  },
  "conversation_metadata": [
    {
      "source_pdf": "cardiology_textbook.pdf",
      "page_number": 45,
      "confidence_score": 0.92,
      "medical_context": "cardiovascular disease"
    }
  ]
}
```

---

## 🔧 Advanced Usage

### 🎛️ **Custom Prompts**

You can modify the generation prompts in `src/ai_models.py`:

```python
def _get_generation_prompt(self, medical_text: str, num_conversations: int) -> str:
    return f"""You are a medical AI assistant creating educational conversation examples...
    
    CUSTOM GUIDELINES:
    - Focus on {your_specific_domain}
    - Include {your_requirements}
    - Ensure {your_quality_criteria}
    """
```

### 📈 **Batch Processing**

```python
from src.dataset_generator import MedicalDatasetGenerator

async def batch_generate():
    generator = MedicalDatasetGenerator()
    
    directories = ["./cardiology", "./neurology", "./pediatrics"]
    
    for directory in directories:
        output_path = await generator.generate_dataset(
            input_dir=Path(directory),
            output_filename=f"{directory.name}_dataset.json",
            max_conversations=1000
        )
        print(f"Generated: {output_path}")

# Run with: python -c "import asyncio; asyncio.run(batch_generate())"
```

### 🔄 **Model Switching**

```python
from src.ai_models import AIModelManager

# Initialize with custom provider
config.api_provider = "openai"
config.default_model = "gpt-4"

manager = AIModelManager()
conversations = await manager.generate_conversations(text, 5)
```

---

## 🐛 Troubleshooting

### ❌ **Common Issues**

#### **API Key Problems**
```bash
# Check configuration
python main.py config-check

# Verify API keys are set
echo $DEEPSEEK_API_KEY
echo $OPENAI_API_KEY
```

#### **Model Loading Issues**
```bash
# Test local model loading
python main.py local-models microsoft/DialoGPT-medium

# Check available models
python main.py local-models --list-models
```

#### **Memory Issues**
```env
# Reduce memory usage
LOAD_IN_8BIT=true
MAX_MEMORY_GB=4
BATCH_SIZE=5
```

#### **Quality Issues**
```env
# Improve quality
TEMPERATURE=0.5
MAX_CONVERSATIONS_PER_CHUNK=3
MIN_CONVERSATION_LENGTH=100
```

### 🔍 **Debug Mode**

```bash
# Enable debug logging
LOG_LEVEL=DEBUG python main.py generate --sample

# Analyze PDF extraction
python main.py analyze --input-dir ./problematic_pdfs
```

### 📞 **Getting Help**

1. **Check Issues**: [GitHub Issues](https://github.com/Ammar-Alnagar/Dataset-Creator/issues)
2. **Read Logs**: Check console output for detailed error messages
3. **Test Configuration**: Run `python main.py config-check`
4. **Start Small**: Use `--sample` flag for testing

---

## 🤝 Contributing

We welcome contributions! Here's how to get started:

### 🛠️ **Development Setup**

```bash
# Clone and setup
git clone https://github.com/Ammar-Alnagar/Dataset-Creator
cd Dataset-Creator
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Install development dependencies
pip install black isort flake8 pytest

# Run tests
pytest

# Format code
black src/
isort src/
```

### 📝 **Contributing Guidelines**

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Ensure code passes linting (`black`, `isort`, `flake8`)
6. Commit changes (`git commit -m 'Add amazing feature'`)
7. Push to branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **DeepSeek** for providing excellent AI capabilities
- **OpenRouter** for democratizing access to AI models  
- **Hugging Face** for transformers and model hosting
- **Medical community** for open-source medical literature

---

## 📊 Project Status

- ✅ **Production Ready**: Stable API and core functionality
- 🔄 **Active Development**: Regular updates and improvements
- 🧪 **Beta Features**: OpenAI-compatible API support
- 📋 **Planned**: GUI interface, more output formats

---

**⭐ Star this repository if you find it useful!**

For questions, issues, or feature requests, please [open an issue](https://github.com/Ammar-Alnagar/Dataset-Creator/issues) on GitHub. 