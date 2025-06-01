#!/usr/bin/env python3
"""Medical Dataset Creator - CLI Interface."""

import asyncio
import sys
from pathlib import Path
from typing import Optional
import functools

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from loguru import logger

from config import config
from src.dataset_generator import MedicalDatasetGenerator


console = Console()


def setup_logging(level: str = "INFO"):
    """Setup logging configuration."""
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )


def run_async_command(func):
    """Wrapper to run async commands in the CLI."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return asyncio.run(func(*args, **kwargs))
    return wrapper


@click.group(invoke_without_command=True)
@click.option('--log-level', default='INFO', help='Set logging level')
@click.version_option(version='1.0.0')
@click.pass_context
def cli(ctx, log_level):
    """Medical Dataset Creator - Generate conversation datasets from medical PDFs.
    
    🏥 INTERACTIVE MODE (Default):
    Simply run 'python main.py' to launch the interactive TUI interface.
    
    📋 AVAILABLE FORMATS:
    • ChatML: {"conversations": [{"from": "human", "value": "..."}]}
    • Instruction: {"instruction": "...", "input": "...", "output": "..."}
    
    🤖 SUPPORTED PROVIDERS:
    • DeepSeek API
    • OpenAI Compatible APIs (OpenRouter, etc.)
    • Local Hugging Face Models
    
    🤗 HUGGING FACE INTEGRATION:
    • Automatic dataset uploads to Hugging Face Hub
    • Use 'hf-auth' to test authentication and configure uploads
    • Use 'hf-upload' to manually upload existing datasets
    
    Use 'python main.py COMMAND --help' for command-specific help.
    """
    setup_logging(log_level)
    
    # If no command is provided, launch TUI
    if ctx.invoked_subcommand is None:
        console.print("[bold green]🏥 Medical Dataset Creator[/bold green]")
        console.print("[yellow]Launching interactive TUI interface...[/yellow]")
        console.print("[dim]Use 'python main.py --help' to see all available commands[/dim]")
        launch_tui()


@cli.command()
def tui():
    """🖥️  Launch the interactive TUI (Terminal User Interface).
    
    The TUI provides a user-friendly menu system where you can:
    • Select AI provider (DeepSeek, OpenAI, Local HuggingFace)
    • Choose dataset format (ChatML or Instruction)
    • Configure generation options (sample vs full, limits, etc.)
    • Monitor generation progress in real-time
    """
    launch_tui()


def launch_tui():
    """Launch the TUI interface."""
    try:
        from src.tui_menu import run_tui
        run_tui()
    except ImportError as e:
        console.print(f"[red]Error: Could not import TUI module. Make sure 'textual' is installed.[/red]")
        console.print(f"[yellow]Run: pip install textual[/yellow]")
        console.print(f"[red]Details: {e}[/red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error launching TUI: {e}[/red]")
        sys.exit(1)


async def _generate_async(input_dir, output_file, max_conversations, sample, use_local):
    """Generate a medical conversation dataset from PDF files."""
    
    try:
        # Temporarily override local model setting if requested
        original_use_local = config.use_local_model
        if use_local:
            config.use_local_model = True
        
        generator = MedicalDatasetGenerator()
        
        # Convert input_dir to Path if provided
        input_path = Path(input_dir) if input_dir else None
        
        with console.status("[bold green]Generating dataset...") as status:
            if sample:
                console.print("[yellow]Generating sample dataset (50 conversations)...")
                output_path = await generator.generate_sample_dataset()
            else:
                console.print("[yellow]Starting full dataset generation...")
                output_path = await generator.generate_dataset(
                    input_dir=input_path,
                    output_filename=output_file,
                    max_conversations=max_conversations
                )
        
        # Display results
        stats = generator.get_statistics()
        display_results(output_path, stats)
        
        # Restore original setting
        config.use_local_model = original_use_local
        
    except Exception as e:
        console.print(f"[red]Error: {e}")
        sys.exit(1)


@cli.command()
@click.option('--input-dir', '-i', type=click.Path(exists=True, file_okay=False, dir_okay=True), 
              help='Directory containing PDF files')
@click.option('--output-file', '-o', help='Output filename for the dataset')
@click.option('--max-conversations', '-m', type=int, help='Maximum number of conversations to generate')
@click.option('--sample', is_flag=True, help='Generate a small sample dataset (50 conversations)')
@click.option('--use-local', is_flag=True, help='Force use of local Hugging Face model')
@run_async_command
def generate(input_dir, output_file, max_conversations, sample, use_local):
    """Generate a medical conversation dataset from PDF files."""
    return _generate_async(input_dir, output_file, max_conversations, sample, use_local)


@cli.command()
@click.option('--input-dir', '-i', type=click.Path(exists=True, file_okay=False, dir_okay=True),
              help='Directory containing PDF files')
def analyze(input_dir):
    """Analyze PDF files and show extraction statistics without generating conversations."""
    
    try:
        from src.pdf_processor import PDFProcessor
        
        input_path = Path(input_dir) if input_dir else config.input_dir
        
        console.print(f"[yellow]Analyzing PDFs in {input_path}...")
        
        processor = PDFProcessor()
        chunks = processor.process_directory(input_path)
        
        if not chunks:
            console.print("[red]No PDF files found or no text could be extracted.")
            return
        
        # Create analysis table
        table = Table(title="PDF Analysis Results")
        table.add_column("PDF File", style="cyan")
        table.add_column("Chunks", justify="right", style="magenta")
        table.add_column("Total Characters", justify="right", style="green")
        table.add_column("Avg Confidence", justify="right", style="yellow")
        
        total_chunks = 0
        total_chars = 0
        
        for pdf_name, pdf_chunks in chunks.items():
            chunk_count = len(pdf_chunks)
            char_count = sum(len(chunk.content) for chunk in pdf_chunks)
            avg_confidence = sum(chunk.confidence_score for chunk in pdf_chunks) / chunk_count
            
            table.add_row(
                pdf_name,
                str(chunk_count),
                f"{char_count:,}",
                f"{avg_confidence:.2f}"
            )
            
            total_chunks += chunk_count
            total_chars += char_count
        
        console.print(table)
        console.print(f"\n[bold]Summary:[/bold]")
        console.print(f"Total PDFs: {len(chunks)}")
        console.print(f"Total chunks: {total_chunks}")
        console.print(f"Total characters: {total_chars:,}")
        console.print(f"Estimated conversations: {total_chunks * config.max_conversations_per_chunk}")
        
    except Exception as e:
        console.print(f"[red]Error: {e}")
        sys.exit(1)


@cli.command()
def config_check():
    """Check configuration and available AI models."""
    
    console.print("[bold]Configuration Check[/bold]\n")
    
    # Display configuration
    config_table = Table(title="Configuration Settings")
    config_table.add_column("Setting", style="cyan")
    config_table.add_column("Value", style="green")
    
    config_table.add_row("API Provider", config.api_provider)
    config_table.add_row("Input Directory", str(config.input_dir))
    config_table.add_row("Output Directory", str(config.output_dir))
    config_table.add_row("Models Cache Directory", str(config.models_cache_dir))
    config_table.add_row("Default Model", config.default_model)
    config_table.add_row("Local Model Name", config.local_model_name)
    config_table.add_row("Use Local Model", str(config.use_local_model))
    config_table.add_row("Temperature", str(config.temperature))
    config_table.add_row("Max Tokens", str(config.max_tokens))
    config_table.add_row("Chunk Size", str(config.chunk_size))
    config_table.add_row("Max Conversations/Chunk", str(config.max_conversations_per_chunk))
    config_table.add_row("Device", config.device)
    config_table.add_row("Load in 8-bit", str(config.load_in_8bit))
    
    console.print(config_table)
    
    # Check API keys and hardware
    console.print("\n[bold]System Status[/bold]")
    
    status_table = Table()
    status_table.add_column("Component", style="cyan")
    status_table.add_column("Status", style="green")
    
    # API Keys
    deepseek_status = "✅ Configured" if config.deepseek_api_key else "❌ Not configured"
    status_table.add_row("DeepSeek API Key", deepseek_status)
    
    openai_status = "✅ Configured" if config.openai_api_key else "❌ Not configured"
    status_table.add_row("OpenAI API Key", openai_status)
    
    hf_status = "✅ Configured" if config.huggingface_api_key else "❌ Not configured"
    status_table.add_row("Hugging Face API Key", hf_status)
    
    if config.openai_base_url:
        status_table.add_row("OpenAI Base URL", config.openai_base_url)
    
    # Legacy Gemini support
    if config.gemini_api_key:
        status_table.add_row("Gemini API Key (Deprecated)", "⚠️ Configured but deprecated")
    
    # Hardware
    try:
        import torch
        if torch.cuda.is_available():
            gpu_status = f"✅ CUDA available ({torch.cuda.get_device_name(0)})"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            gpu_status = "✅ MPS (Apple Silicon) available"
        else:
            gpu_status = "⚠️  CPU only"
        status_table.add_row("Hardware Acceleration", gpu_status)
        
        memory_info = f"{torch.cuda.get_device_properties(0).total_memory // 1024**3} GB" if torch.cuda.is_available() else "N/A"
        status_table.add_row("GPU Memory", memory_info)
    except ImportError:
        status_table.add_row("PyTorch", "❌ Not installed")
    
    console.print(status_table)
    
    # Test model availability
    try:
        from src.ai_models import AIModelManager
        
        console.print("\n[bold]Testing Model Availability...[/bold]")
        manager = AIModelManager()
        available_models = manager.get_available_models()
        
        if available_models:
            console.print(f"[green]✅ Available models: {', '.join(available_models)}")
        else:
            console.print("[red]❌ No models available. Please configure API keys or enable local models.")
            
    except Exception as e:
        console.print(f"[red]❌ Error testing models: {e}")


@cli.command()
@click.argument('model_name', default='')
@click.option('--list-models', is_flag=True, help='List available local models')
def local_models(model_name, list_models):
    """Manage local Hugging Face models."""
    
    if list_models:
        console.print("[bold]Popular Medical/Conversational Models:[/bold]")
        
        models_table = Table()
        models_table.add_column("Model Name", style="cyan")
        models_table.add_column("Type", style="green")
        models_table.add_column("Size", style="yellow")
        models_table.add_column("Description", style="white")
        
        recommended_models = [
            ("microsoft/DialoGPT-medium", "Conversational", "~1.5GB", "Good for dialogue generation"),
            ("microsoft/DialoGPT-large", "Conversational", "~3GB", "Better quality dialogue"),
            ("facebook/blenderbot-400M-distill", "Conversational", "~1.6GB", "Optimized for conversation"),
            ("google/flan-t5-base", "Text-to-Text", "~1GB", "Instruction-following model"),
            ("google/flan-t5-large", "Text-to-Text", "~3GB", "Larger instruction model"),
            ("allenai/led-base-16384", "Medical", "~800MB", "Scientific/medical text"),
        ]
        
        for model, type_str, size, desc in recommended_models:
            models_table.add_row(model, type_str, size, desc)
        
        console.print(models_table)
        console.print(f"\n[bold]Usage:[/bold] medical-dataset-creator local-models <model_name>")
        console.print(f"[bold]Example:[/bold] medical-dataset-creator local-models microsoft/DialoGPT-medium")
        return
    
    if not model_name:
        console.print("[red]Please provide a model name or use --list-models")
        return
    
    try:
        console.print(f"[yellow]Testing local model: {model_name}")
        
        from src.ai_models import HuggingFaceLocalModel
        
        with console.status("[bold green]Loading model..."):
            model = HuggingFaceLocalModel(model_name)
        
        if model.is_available():
            console.print(f"[green]✅ Model {model_name} loaded successfully!")
            console.print(f"[green]📁 Cached in: {config.models_cache_dir}")
            console.print(f"[green]🔧 Device: {model.device}")
        else:
            console.print(f"[red]❌ Failed to load model {model_name}")
            
    except Exception as e:
        console.print(f"[red]Error loading model: {e}")


@cli.command()
@click.argument('dataset_file', type=click.Path(exists=True))
def validate(dataset_file):
    """Validate a generated dataset file."""
    
    try:
        import json
        
        console.print(f"[yellow]Validating dataset: {dataset_file}")
        
        with open(dataset_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Check if data has the conversations wrapper
        if isinstance(data, dict) and "conversations" in data:
            conversations_data = data["conversations"]
            console.print("[green]✅ Found 'conversations' wrapper")
        elif isinstance(data, list):
            conversations_data = data
            console.print("[yellow]⚠️  Old format detected (direct array)")
        else:
            console.print("[red]❌ Invalid dataset format")
            return
        
        if not isinstance(conversations_data, list):
            console.print("[red]❌ Conversations should be a list")
            return
        
        valid_conversations = 0
        total_conversations = len(conversations_data)
        
        for i, conversation in enumerate(conversations_data):
            if not isinstance(conversation, list) or len(conversation) != 2:
                console.print(f"[red]❌ Invalid conversation format at index {i}")
                continue
            
            user_msg = conversation[0]
            assistant_msg = conversation[1]
            
            if (user_msg.get("from") != "user" or 
                assistant_msg.get("from") != "assistant" or
                not user_msg.get("value") or 
                not assistant_msg.get("value")):
                console.print(f"[red]❌ Invalid message format at index {i}")
                continue
            
            valid_conversations += 1
        
        # Display validation results
        console.print(f"\n[bold]Validation Results:[/bold]")
        console.print(f"Total conversations: {total_conversations}")
        console.print(f"Valid conversations: {valid_conversations}")
        console.print(f"Invalid conversations: {total_conversations - valid_conversations}")
        
        if valid_conversations == total_conversations:
            console.print("[green]✅ Dataset is valid!")
        else:
            console.print(f"[yellow]⚠️  {total_conversations - valid_conversations} conversations have issues")
        
    except Exception as e:
        console.print(f"[red]Error validating dataset: {e}")


@cli.command()
def hf_auth():
    """Test Hugging Face authentication and show upload settings."""
    
    from src.huggingface_uploader import HuggingFaceUploader
    
    console.print("[bold]🤗 Hugging Face Configuration[/bold]\n")
    
    # Show current settings
    settings_table = Table()
    settings_table.add_column("Setting", style="cyan")
    settings_table.add_column("Value", style="green")
    
    hf_key_status = "✅ Configured" if config.huggingface_api_key else "❌ Not configured"
    settings_table.add_row("API Key", hf_key_status)
    settings_table.add_row("Upload Enabled", "✅ Yes" if config.enable_hf_upload else "❌ No")
    repo_name_display = config.hf_repo_name or "🔄 Auto-generated from filename"
    settings_table.add_row("Repository Name", repo_name_display)
    settings_table.add_row("Private Repository", "✅ Yes" if config.hf_dataset_private else "❌ No")
    settings_table.add_row("Commit Message", config.hf_commit_message)
    
    console.print(settings_table)
    
    # Test authentication if API key is configured
    if config.huggingface_api_key:
        console.print("\n[bold]Testing Authentication...[/bold]")
        
        try:
            uploader = HuggingFaceUploader()
            if uploader.authenticate():
                console.print("[green]✅ Authentication successful!")
            else:
                console.print("[red]❌ Authentication failed")
        except Exception as e:
            console.print(f"[red]❌ Authentication error: {e}")
    else:
        console.print("\n[yellow]⚠️  To configure Hugging Face uploads:[/yellow]")
        console.print("1. Get your API key from: https://huggingface.co/settings/tokens")
        console.print("2. Add HUGGINGFACE_API_KEY=your_key_here to your .env file")
        console.print("3. Set ENABLE_HF_UPLOAD=true to enable automatic uploads")
        console.print("4. (Optional) Set HF_REPO_NAME=username/dataset-name for custom repo names")
        console.print("   Otherwise, repository names will be auto-generated from filenames")


@cli.command()
@click.argument('dataset_file', type=click.Path(exists=True))
@click.option('--repo-name', '-r', help='Hugging Face repository name (username/dataset-name)')
@click.option('--private', is_flag=True, help='Make the repository private')
@click.option('--commit-message', '-m', help='Custom commit message')
def hf_upload(dataset_file, repo_name, private, commit_message):
    """Upload an existing dataset file to Hugging Face Hub."""
    
    from src.huggingface_uploader import HuggingFaceUploader
    from pathlib import Path
    
    console.print(f"[bold]🤗 Uploading dataset to Hugging Face Hub[/bold]\n")
    
    dataset_path = Path(dataset_file)
    
    try:
        uploader = HuggingFaceUploader()
        
        # Authenticate
        if not uploader.authenticate():
            console.print("[red]❌ Failed to authenticate with Hugging Face")
            return
        
        # Use provided parameters or config defaults
        repo_name = repo_name or config.hf_repo_name
        private = private if private is not None else config.hf_dataset_private
        commit_message = commit_message or config.hf_commit_message
        
        if not repo_name:
            console.print("[yellow]ℹ️  No repository name specified - will auto-generate from filename[/yellow]")
        
        console.print(f"[yellow]Uploading to: {repo_name or 'Auto-generated'}[/yellow]")
        console.print(f"[yellow]Private: {private}[/yellow]")
        console.print(f"[yellow]Commit message: {commit_message}[/yellow]")
        
        with console.status("[bold green]Uploading dataset..."):
            success = uploader.upload_dataset(
                dataset_path=dataset_path,
                repo_name=repo_name,
                private=private,
                commit_message=commit_message
            )
        
        if success:
            console.print(f"[green]✅ Successfully uploaded dataset!")
            console.print(f"[green]🔗 View at: https://huggingface.co/datasets/{repo_name}")
        else:
            console.print("[red]❌ Upload failed")
            
    except Exception as e:
        console.print(f"[red]❌ Upload error: {e}")


def display_results(output_path: Path, stats):
    """Display generation results in a nice format."""
    
    console.print("\n[bold green]🎉 Dataset Generation Complete![/bold green]\n")
    
    # Results table
    results_table = Table(title="Generation Results")
    results_table.add_column("Metric", style="cyan")
    results_table.add_column("Value", style="green")
    
    results_table.add_row("Output File", str(output_path))
    results_table.add_row("PDFs Processed", str(stats.total_pdfs_processed))
    results_table.add_row("Text Chunks", str(stats.total_chunks_extracted))
    results_table.add_row("Conversations Generated", str(stats.total_conversations_generated))
    results_table.add_row("Success Rate", f"{(stats.successful_generations / stats.total_chunks_extracted * 100):.1f}%")
    results_table.add_row("Average Quality Score", f"{stats.average_confidence_score:.2f}")
    results_table.add_row("Processing Time", f"{stats.processing_time_seconds:.1f}s")
    results_table.add_row("Models Used", ", ".join(stats.models_used))
    
    console.print(results_table)
    
    # Show upload status if enabled
    if config.enable_hf_upload and config.hf_repo_name:
        console.print(f"\n[bold green]🤗 Dataset uploaded to Hugging Face:[/bold green]")
        console.print(f"[green]🔗 https://huggingface.co/datasets/{config.hf_repo_name}")
    elif config.enable_hf_upload:
        console.print(f"\n[yellow]⚠️  Hugging Face upload enabled but repository name not set")
    
    # Quick preview
    console.print(f"\n[bold]Dataset saved to:[/bold] {output_path}")
    console.print(f"[bold]Metadata saved to:[/bold] {output_path.with_suffix('.metadata.json')}")


if __name__ == '__main__':
    cli() 