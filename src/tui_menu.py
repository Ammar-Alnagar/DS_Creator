"""TUI Menu for Medical Dataset Creator."""

import asyncio
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import (
    Button, 
    Header, 
    Footer, 
    Static, 
    Select, 
    Input, 
    Checkbox,
    TextArea,
    ProgressBar
)
from textual.screen import Screen
from loguru import logger

from config import config
from src.dataset_generator import MedicalDatasetGenerator


@dataclass
class GenerationConfig:
    """Configuration for dataset generation."""
    provider: str
    format_type: str
    use_sample: bool
    max_conversations: Optional[int]
    input_dir: Optional[Path]
    output_filename: Optional[str]
    sample_count: int


class GenerationScreen(Screen):
    """Screen for running the dataset generation."""
    
    def __init__(self, generation_config: GenerationConfig):
        super().__init__()
        self.generation_config = generation_config
        self.generator = MedicalDatasetGenerator()
        self.stop_requested = asyncio.Event()
        
    def compose(self) -> ComposeResult:
        yield Header()
        yield Container(
            Static("🏥 Medical Dataset Generation", classes="title"),
            Static("Running generation with your selected configuration...", classes="subtitle"),
            Vertical(
                Static(f"Provider: {self.generation_config.provider}", classes="config-item"),
                Static(f"Format: {self.generation_config.format_type}", classes="config-item"),
                Static(f"Type: {'Sample (' + str(self.generation_config.sample_count) + ' conversations)' if self.generation_config.use_sample else 'Full PDF'}", classes="config-item"),
                Static(f"Max Conversations: {self.generation_config.max_conversations or 'No limit'}", classes="config-item"),
                classes="config-display"
            ),
            ProgressBar(total=100, show_eta=True, classes="progress"),
            TextArea(classes="log-area"),
            Horizontal(
                Button("Cancel", variant="error", id="cancel"),
                Button("Back to Menu", variant="primary", id="back", disabled=True),
                classes="button-row"
            ),
            classes="generation-container"
        )
        yield Footer()
    
    async def on_mount(self):
        """Start generation when screen mounts."""
        progress_bar = self.query_one(ProgressBar)
        log_area = self.query_one(TextArea)
        back_button = self.query_one("#back", Button)
        cancel_button = self.query_one("#cancel", Button)
        
        log_area.read_only = True
        
        # Pass the stop_requested event to the generator
        self.generator.set_stop_event(self.stop_requested)

        try:
            # Update config based on selection
            original_provider = config.api_provider
            config.api_provider = self.generation_config.provider
            
            # Set dataset format
            config.dataset_format = self.generation_config.format_type
            
            log_area.text = "Starting dataset generation...\n"
            progress_bar.update(progress=10)
            
            if self.generation_config.use_sample:
                log_area.text += f"Generating sample dataset ({self.generation_config.sample_count} conversations)...\n"
                progress_bar.update(progress=30)
                output_path = await self.generator.generate_sample_dataset(self.generation_config.sample_count)
            else:
                log_area.text += "Generating full dataset...\n"
                progress_bar.update(progress=30)
                output_path = await self.generator.generate_dataset(
                    input_dir=self.generation_config.input_dir,
                    output_filename=self.generation_config.output_filename,
                    max_conversations=self.generation_config.max_conversations
                )
            
            progress_bar.update(progress=100)
            
            # Get stats
            stats = self.generator.get_statistics()
            log_area.text += f"\n✅ Generation completed successfully!\n"
            log_area.text += f"📁 Output file: {output_path}\n"
            log_area.text += f"📊 Conversations generated: {stats.total_conversations_generated}\n"
            log_area.text += f"⏱️  Processing time: {stats.processing_time_seconds:.1f} seconds\n"
            log_area.text += f"🎯 Success rate: {((stats.successful_generations / stats.total_chunks_extracted) * 100):.1f}%\n"
            
            # Restore original config
            config.api_provider = original_provider
            
        except Exception as e:
            progress_bar.update(progress=0)
            if self.stop_requested.is_set():
                log_area.text += f"\n🛑 Generation stopped by user.\n"
                logger.info("Generation stopped by user.")
            else:
                log_area.text += f"\n❌ Error during generation: {str(e)}\n"
                logger.error(f"Generation error: {e}")
        
        finally:
            back_button.disabled = False
            cancel_button.disabled = True # Disable cancel if process finished or stopped
            if self.stop_requested.is_set():
                cancel_button.label = "Stopped"
            else:
                cancel_button.label = "Cancel" # Or keep as "Cancel" if it was not clicked
    
    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel":
            self.stop_requested.set()
            event.button.label = "Stopping..."
            event.button.disabled = True
            # The generator will see this event and stop
            log_area = self.query_one(TextArea)
            log_area.text += "\nAttempting to stop generation...\n"
        elif event.button.id == "back":
            self.app.pop_screen()


class MedicalDatasetCreatorTUI(App):
    """Main TUI application for Medical Dataset Creator."""
    
    CSS = """
    $border: solid #000000;

    .title {
        text-align: center;
        text-style: bold;
        color: $primary;
        margin: 1 0;
    }
    
    .subtitle {
        text-align: center;
        color: $text-muted;
        margin: 0 0 2 0;
    }
    
    .section {
        border: solid $border;
        margin: 1 0;
        padding: 1;
    }
    
    .section-title {
        text-style: bold;
        color: $accent;
        margin: 0 0 1 0;
    }
    
    .button-row {
        align: center middle;
        margin: 2 0 1 0;
    }
    
    .config-display {
        border: solid $border;
        margin: 1 0;
        padding: 1;
    }
    
    .config-item {
        margin: 0 0 0 2;
        color: $text;
    }
    
    .generation-container {
        margin: 1;
        padding: 1;
    }
    
    .progress {
        margin: 1 0;
    }
    
    .log-area {
        height: 15;
        margin: 1 0;
    }
    
    Select {
        margin: 0 0 1 0;
    }
    
    Input {
        margin: 0 0 1 0;
    }
    
    Button {
        margin: 0 1;
    }
    
    Static {
        color: $text;
    }
    """
    
    def __init__(self):
        super().__init__()
        self.title = "Medical Dataset Creator"
        self.sub_title = "Generate conversation datasets from medical PDFs"
        
        # Available options
        self.providers = [
            ("deepseek", "DeepSeek API"),
            ("openai", "OpenAI Compatible API"),
            ("huggingface_local", "Local Hugging Face Model")
        ]
        
        self.formats = [
            ("chatml", "ChatML Format (conversations array)"),
            ("instruction", "Instruction Format (instruction/input/output)")
        ]
        
        # Selection state
        self.selected_provider = "deepseek"
        self.selected_format = "chatml"
        self.use_sample = True
        self.sample_count = 50
        self.max_conversations = None
        self.input_directory = None
        self.output_filename = None
    
    def compose(self) -> ComposeResult:
        yield Header()
        yield Container(
            Static("🏥 Medical Dataset Creator"),
            Static("Configure your dataset generation settings"),
            
            # Provider Selection
            Static("1. Select AI Provider"),
            Select(
                options=[(label, value) for value, label in self.providers],
                value=self.selected_provider,
                id="provider-select"
            ),
            Static("Choose the AI provider for generating conversations"),
            
            # Format Selection
            Static("2. Select Dataset Format"),
            Select(
                options=[(label, value) for value, label in self.formats],
                value=self.selected_format,
                id="format-select"
            ),
            Static("ChatML or Instruction format selection"),
            
            # Generation Options
            Static("3. Generation Options"),
            Checkbox("Generate sample dataset", value=True, id="sample-checkbox"),
            Input(placeholder="Number of samples (e.g., 50)", value="50", id="sample-count-input"),
            Input(placeholder="Max conversations (leave empty for no limit)", id="max-conversations", disabled=True),
            Input(placeholder="Input directory (leave empty for default)", id="input-dir"),
            Input(placeholder="Output filename (auto-generated if empty)", id="output-filename"),
            
            # Configuration Preview
            Static("4. Current Configuration"),
            Static(id="config-preview"),
            
            # Action Buttons
            Horizontal(
                Button("Generate Dataset", variant="success", id="generate"),
                Button("Check Configuration", variant="primary", id="check-config"),
                Button("Exit", variant="error", id="exit"),
            )
        )
        yield Footer()
    
    def on_mount(self):
        """Update preview when app starts."""
        self.update_config_preview()
    
    def on_select_changed(self, event: Select.Changed) -> None:
        """Handle selection changes."""
        if event.select.id == "provider-select":
            self.selected_provider = event.value
        elif event.select.id == "format-select":
            self.selected_format = event.value
        
        self.update_config_preview()
    
    def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        """Handle checkbox changes."""
        if event.checkbox.id == "sample-checkbox":
            self.use_sample = event.value
            sample_count_input = self.query_one("#sample-count-input", Input)
            max_conv_input = self.query_one("#max-conversations", Input)
            
            sample_count_input.disabled = not event.value
            max_conv_input.disabled = event.value
            
            if event.value: # Sample mode is on
                # Ensure max_conversations input is cleared or ignored for sample mode
                max_conv_input.value = ""
                self.max_conversations = None 
            else: # Sample mode is off, enable max_conversations if needed
                pass # Max conversations input is now enabled, user can type
        
        self.update_config_preview()
    
    def on_input_changed(self, event: Input.Changed) -> None:
        """Handle input changes."""
        if event.input.id == "max-conversations":
            try:
                self.max_conversations = int(event.value) if event.value.strip() else None
            except ValueError:
                self.max_conversations = None
        elif event.input.id == "sample-count-input":
            try:
                value = int(event.value) if event.value.strip() else 50 # Default to 50 if empty
                self.sample_count = max(1, value) # Ensure at least 1 sample
            except ValueError:
                self.sample_count = 50 # Default on error
            # Update the input field to reflect sanitized value
            event.input.value = str(self.sample_count)
        elif event.input.id == "input-dir":
            self.input_directory = Path(event.value) if event.value.strip() else None
        elif event.input.id == "output-filename":
            self.output_filename = event.value.strip() if event.value.strip() else None
        
        self.update_config_preview()
    
    def update_config_preview(self):
        """Update the configuration preview."""
        preview = self.query_one("#config-preview", Static)
        
        provider_name = next((label for value, label in self.providers if value == self.selected_provider), "Unknown")
        format_name = next((label for value, label in self.formats if value == self.selected_format), "Unknown")
        
        if self.use_sample:
            config_text = f"""Provider: {provider_name}
Format: {format_name}
Mode: Sample ({self.sample_count} conversations)"""
        else:
            config_text = f"""Provider: {provider_name}
Format: {format_name}
Mode: Full PDF processing"""
        
        if not self.use_sample and self.max_conversations:
            config_text += f"\nMax Conversations: {self.max_conversations}"
        
        if self.input_directory:
            config_text += f"\nInput Directory: {self.input_directory}"
        else:
            config_text += f"\nInput Directory: {config.input_dir} (default)"
        
        if self.output_filename:
            config_text += f"\nOutput Filename: {self.output_filename}"
        else:
            config_text += "\nOutput Filename: Auto-generated"
        
        preview.update(config_text)
    
    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "exit":
            self.exit()
        
        elif event.button.id == "check-config":
            await self.check_configuration()
        
        elif event.button.id == "generate":
            await self.start_generation()
    
    async def check_configuration(self):
        """Check if the current configuration is valid."""
        try:
            from src.ai_models import AIModelManager
            
            # Check API keys and model availability
            manager = AIModelManager()
            
            # Temporarily set provider
            original_provider = config.api_provider
            config.api_provider = self.selected_provider
            
            available_models = manager.get_available_models()
            
            if available_models:
                self.notify(f"✅ Configuration valid! Models available: {', '.join(available_models)}")
            else:
                self.notify(f"❌ No models available for provider '{self.selected_provider}'. Check API keys.", severity="error")
            
            # Restore original provider
            config.api_provider = original_provider
            
        except Exception as e:
            self.notify(f"❌ Configuration error: {str(e)}", severity="error")
    
    async def start_generation(self):
        """Start the dataset generation process."""
        # Validate configuration
        if self.selected_provider == "deepseek" and not config.deepseek_api_key:
            self.notify("❌ DeepSeek API key not configured!", severity="error")
            return
        
        if self.selected_provider == "openai" and not config.openai_api_key:
            self.notify("❌ OpenAI API key not configured!", severity="error")
            return
        
        # Create generation config
        generation_config = GenerationConfig(
            provider=self.selected_provider,
            format_type=self.selected_format,
            use_sample=self.use_sample,
            sample_count=self.sample_count if self.use_sample else 0,
            max_conversations=self.max_conversations,
            input_dir=self.input_directory or config.input_dir,
            output_filename=self.output_filename
        )
        
        # Push generation screen
        self.push_screen(GenerationScreen(generation_config))


def run_tui():
    """Run the TUI application."""
    app = MedicalDatasetCreatorTUI()
    app.run()


if __name__ == "__main__":
    run_tui() 