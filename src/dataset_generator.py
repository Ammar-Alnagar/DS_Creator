"""Main dataset generator orchestrating the entire pipeline."""

import json
import asyncio
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

from loguru import logger
from rich.progress import Progress

from config import config
from src.pdf_processor import PDFProcessor, TextChunk
from src.ai_models import AIModelManager, ConversationPair
from src.huggingface_uploader import HuggingFaceUploader


@dataclass
class DatasetEntry:
    """Represents a single entry in the final dataset."""
    
    user_message: str
    assistant_message: str
    metadata: Dict[str, Any]


@dataclass
class GenerationStats:
    """Statistics about the dataset generation process."""
    
    total_pdfs_processed: int
    total_chunks_extracted: int
    total_conversations_generated: int
    successful_generations: int
    failed_generations: int
    average_confidence_score: float
    processing_time_seconds: float
    models_used: List[str]


class MedicalDatasetGenerator:
    """Main class for generating medical conversation datasets from PDFs."""
    
    def __init__(self):
        self.pdf_processor = PDFProcessor()
        self.ai_manager = AIModelManager()
        self.hf_uploader = HuggingFaceUploader() if config.enable_hf_upload else None
        self.stats = GenerationStats(
            total_pdfs_processed=0,
            total_chunks_extracted=0,
            total_conversations_generated=0,
            successful_generations=0,
            failed_generations=0,
            average_confidence_score=0.0,
            processing_time_seconds=0.0,
            models_used=[]
        )
        self.stop_event: Optional[asyncio.Event] = None
    
    def set_stop_event(self, stop_event: asyncio.Event):
        """Set the event used to signal a stop request."""
        self.stop_event = stop_event
        # Also pass to AI manager if it needs to be aware of stops during its own long operations
        if hasattr(self.ai_manager, 'set_stop_event'):
            self.ai_manager.set_stop_event(stop_event)

    async def _early_exit(self, conversations: List[DatasetEntry], output_filename: Optional[str], start_time: datetime, hf_repo_name: Optional[str] = None) -> Path:
        """Handle early exit due to stop request or error."""
        logger.info("Generation process is exiting early.")
        
        # Update final stats for partial generation
        self.stats.total_conversations_generated = len(conversations)
        if conversations:
            confidence_scores = [conv.metadata.get('confidence_score', 0.0) for conv in conversations if conv.metadata]
            if confidence_scores:
                 self.stats.average_confidence_score = sum(confidence_scores) / len(confidence_scores)
            else:
                self.stats.average_confidence_score = 0.0
        
        end_time = datetime.now()
        self.stats.processing_time_seconds = (end_time - start_time).total_seconds()
        
        # Save whatever was generated so far
        if not output_filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"medical_conversations_partial_{timestamp}.json"
            
        output_path = await self._save_dataset(conversations, output_filename, hf_repo_name)
        
        self._log_generation_stats()
        logger.info(f"Partial dataset saved to: {output_path}")
        return output_path

    async def generate_dataset(
        self, 
        input_dir: Optional[Path] = None,
        output_filename: Optional[str] = None,
        max_conversations: Optional[int] = None,
        hf_repo_name: Optional[str] = None,
        progress_bar: Optional[Progress] = None,
        main_task_id: Optional[Any] = None
    ) -> Path:
        """
        Generate a complete medical conversation dataset from PDFs.
        
        Args:
            input_dir: Directory containing PDF files (defaults to config.input_dir)
            output_filename: Name for output file (auto-generated if None)
            max_conversations: Maximum number of conversations to generate
            hf_repo_name: Name of the Hugging Face repository to upload the dataset to
            progress_bar: Rich progress bar for tracking progress
            main_task_id: ID of the main task in the progress bar
            
        Returns:
            Path to the generated dataset file
        """
        start_time = datetime.now()
        logger.info("Starting medical dataset generation...")
        
        # Set defaults
        input_dir = input_dir or config.input_dir
        if not output_filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"medical_conversations_{timestamp}.json"
        
        # --- PDF Processing Stage ---
        logger.info(f"Processing PDFs from {input_dir}")
        
        pdf_files_to_process = list(input_dir.glob("*.pdf"))
        if not pdf_files_to_process:
            raise ValueError(f"No PDF files found in {input_dir}")

        pdf_processing_sub_task_id = None
        if progress_bar:
            pdf_processing_sub_task_id = progress_bar.add_task(
                f"Extracting text from {len(pdf_files_to_process)} PDF(s)...", 
                total=len(pdf_files_to_process)
            )
            if main_task_id:
                 progress_bar.update(main_task_id, description="Phase 1: PDF Processing")


        pdf_chunks = self.pdf_processor.process_directory(
            input_dir,
            pdf_files_list=pdf_files_to_process,
            progress_bar=progress_bar,
            task_id=pdf_processing_sub_task_id
        )
        
        if progress_bar and pdf_processing_sub_task_id:
            completed_count = progress_bar.tasks[progress_bar.task_ids.index(pdf_processing_sub_task_id)].completed
            total_count = progress_bar.tasks[progress_bar.task_ids.index(pdf_processing_sub_task_id)].total
            progress_bar.update(pdf_processing_sub_task_id, 
                                description=f"Processed {completed_count}/{total_count} PDFs.",
                                completed=total_count)

        if not pdf_chunks:
            processed_pdf_count = sum(1 for _ in pdf_chunks.values() if _) 
            if processed_pdf_count == 0:
                 raise ValueError(f"No usable text extracted from any PDFs in {input_dir}")

        # Update stats
        self.stats.total_pdfs_processed = len(pdf_chunks)
        self.stats.total_chunks_extracted = sum(len(chunks) for chunks in pdf_chunks.values())
        self.stats.models_used = self.ai_manager.get_available_models()
        
        logger.info(f"Extracted {self.stats.total_chunks_extracted} chunks from {self.stats.total_pdfs_processed} PDFs")

        if self.stats.total_chunks_extracted == 0:
            logger.warning("No chunks were extracted. Cannot proceed with conversation generation.")
            output_path = await self._save_dataset([], output_filename, hf_repo_name)
            self._log_generation_stats()
            return output_path
        
        # --- Conversation Generation Stage ---
        all_conversations = []
        
        if progress_bar and main_task_id:
            progress_bar.update(main_task_id, 
                                total=self.stats.total_chunks_extracted, 
                                completed=0, 
                                description=f"Phase 2: Generating Conversations (0/{self.stats.total_chunks_extracted} chunks)")

        processed_chunk_count = 0
        for pdf_name, chunks in pdf_chunks.items():
            pdf_file_name_only = Path(pdf_name).name
            logger.info(f"Processing {len(chunks)} chunks from {pdf_file_name_only}")
            
            if self.stop_event and self.stop_event.is_set():
                logger.info("Stop requested during PDF processing, halting generation.")
                return await self._early_exit(all_conversations, output_filename, start_time, hf_repo_name)

            for chunk_idx, chunk in enumerate(chunks):
                if self.stop_event and self.stop_event.is_set():
                    logger.info("Stop requested during chunk processing, halting generation.")
                    return await self._early_exit(all_conversations, output_filename, start_time, hf_repo_name)
                
                if progress_bar and main_task_id:
                    progress_bar.update(main_task_id, 
                                        description=f"AI: {pdf_file_name_only} ({chunk_idx+1}/{len(chunks)}) | Overall ({processed_chunk_count}/{self.stats.total_chunks_extracted})")
                try:
                    # Generate conversations for this chunk
                    current_chunk_conversations = await self.ai_manager.generate_conversations(
                        chunk.content, 
                        config.max_conversations_per_chunk
                    )
                    
                    if current_chunk_conversations:
                        # Filter and process conversations
                        processed_conversations = self._process_conversations(
                            current_chunk_conversations, chunk, pdf_file_name_only
                        )
                        all_conversations.extend(processed_conversations)
                        self.stats.successful_generations += 1
                    else:
                        self.stats.failed_generations += 1
                        logger.warning(f"No conversations generated for chunk {chunk.chunk_index} from {pdf_file_name_only}")
                    
                    processed_chunk_count +=1
                    if progress_bar and main_task_id:
                        progress_bar.advance(main_task_id)
                    
                    # Check if we've reached the maximum
                    if max_conversations and len(all_conversations) >= max_conversations:
                        logger.info(f"Reached maximum conversations limit: {max_conversations}")
                        all_conversations = all_conversations[:max_conversations]
                        if progress_bar and main_task_id:
                            current_task = next((t for t in progress_bar.tasks if t.id == main_task_id), None)
                            if current_task:
                                progress_bar.update(main_task_id, 
                                                    completed=current_task.total, 
                                                    description=f"Reached max conversations. Processed {processed_chunk_count}/{self.stats.total_chunks_extracted} chunks.")
                        break 
                    
                    # Add delay to respect API rate limits
                    await asyncio.sleep(config.api_request_delay)
                    if self.stop_event and self.stop_event.is_set():
                        logger.info("Stop requested after API call/delay, halting generation.")
                        return await self._early_exit(all_conversations, output_filename, start_time, hf_repo_name)
                        
                except Exception as e:
                    logger.error(f"Error processing chunk {chunk.chunk_index} from {pdf_file_name_only}: {e}")
                    self.stats.failed_generations += 1
                    processed_chunk_count +=1
                    if progress_bar and main_task_id:
                        progress_bar.advance(main_task_id)
                    continue
            
            if max_conversations and len(all_conversations) >= max_conversations:
                break 
        
        # Final update to main task description after loop, if not hit max_conversations
        if progress_bar and main_task_id:
            current_task = next((t for t in progress_bar.tasks if t.id == main_task_id), None)
            if current_task and current_task.completed < current_task.total:
                 progress_bar.update(main_task_id, description=f"Phase 2: Completed Generation ({current_task.completed}/{current_task.total} chunks)")
            elif current_task and current_task.completed == current_task.total:
                 progress_bar.update(main_task_id, description=f"Phase 2: All {current_task.total} chunks processed.")

        # Update final stats
        self.stats.total_conversations_generated = len(all_conversations)
        if all_conversations:
            confidence_scores = [conv.metadata.get('confidence_score', 0.0) for conv in all_conversations]
            self.stats.average_confidence_score = sum(confidence_scores) / len(confidence_scores)
        
        end_time = datetime.now()
        self.stats.processing_time_seconds = (end_time - start_time).total_seconds()
        
        # Save dataset
        output_path = await self._save_dataset(all_conversations, output_filename, hf_repo_name)
        
        # Log final statistics
        self._log_generation_stats()
        
        logger.info(f"Dataset generation completed! Output saved to: {output_path}")
        return output_path
    
    def _process_conversations(
        self, 
        conversations: List[ConversationPair], 
        chunk: TextChunk, 
        pdf_name: str
    ) -> List[DatasetEntry]:
        """Process and filter conversations into dataset entries."""
        
        processed = []
        
        for conv in conversations:
            # Apply quality filters
            if not self._passes_quality_filters(conv):
                continue
            
            # Create dataset entry
            entry = DatasetEntry(
                user_message=conv.user_message,
                assistant_message=conv.assistant_message,
                metadata={
                    'source_pdf': pdf_name,
                    'page_number': chunk.page_number,
                    'chunk_index': chunk.chunk_index,
                    'extraction_method': chunk.extraction_method,
                    'confidence_score': conv.confidence_score,
                    'medical_context': conv.medical_context,
                    'generation_metadata': conv.generation_metadata,
                    'timestamp': datetime.now().isoformat(),
                    'user_message_length': len(conv.user_message),
                    'assistant_message_length': len(conv.assistant_message)
                }
            )
            
            processed.append(entry)
        
        return processed
    
    def _passes_quality_filters(self, conversation: ConversationPair) -> bool:
        """Apply quality filters to conversation pairs."""
        
        # Check minimum confidence score
        if conversation.confidence_score < 0.3:
            return False
        
        # Check message lengths
        user_len = len(conversation.user_message)
        assistant_len = len(conversation.assistant_message)
        
        if user_len < config.min_conversation_length or user_len > config.max_conversation_length:
            return False
        
        if assistant_len < config.min_conversation_length or assistant_len > config.max_conversation_length:
            return False
        
        # Check for basic medical relevance
        medical_keywords = [
            'symptom', 'pain', 'doctor', 'treatment', 'medication', 'health',
            'condition', 'diagnosis', 'therapy', 'medical', 'patient', 'disease'
        ]
        
        combined_text = (conversation.user_message + ' ' + conversation.assistant_message).lower()
        if not any(keyword in combined_text for keyword in medical_keywords):
            return False
        
        return True
    
    async def _save_dataset(self, conversations: List[DatasetEntry], filename: str, hf_repo_name: Optional[str] = None) -> Path:
        """Save the dataset in the required format matching the reference structure."""
        
        output_path = config.output_dir / filename
        
        if config.dataset_format == "instruction":
            # Instruction format: instruction/input/output
            dataset_records = []
            
            for i, entry in enumerate(conversations):
                # For instruction format, we need to transform the conversation
                # Extract medical context as instruction, user message as input, assistant message as output
                medical_context = entry.metadata.get('medical_context', '')
                
                # Create instruction based on medical context or use a default
                if medical_context:
                    instruction = f"You are a medical AI assistant. Based on the following medical context, provide helpful and accurate information: {medical_context[:200]}..."
                else:
                    instruction = "You are a medical AI assistant. Provide helpful, accurate medical information while encouraging users to consult healthcare professionals for specific medical advice."
                
                record = {
                    "instruction": instruction,
                    "input": entry.user_message,
                    "output": entry.assistant_message
                }
                
                dataset_records.append(record)
            
            format_info = {
                'description': 'Instruction-tuning format with instruction, input, and output fields',
                'format_type': 'instruction_tuning',
                'total_records': len(dataset_records),
                'structure': 'Each record contains instruction (system prompt), input (user query), and output (assistant response)'
            }
        
        else:
            # Default ChatML format: conversations array
            dataset_records = []
            
            for i, entry in enumerate(conversations):
                # Create a record for each conversation pair with separate human/gpt messages
                record = {
                    "id": str(i + 1),  # Start IDs from 1 to match reference format
                    "conversations": [
                        {
                            "from": "human",
                            "value": entry.user_message
                        },
                        {
                            "from": "gpt", 
                            "value": entry.assistant_message
                        }
                    ]
                }
                
                dataset_records.append(record)
            
            format_info = {
                'description': 'Each conversation pair is stored as a separate record with ID and conversations array',
                'format_type': 'id_conversations_array',
                'total_conversation_pairs': len(conversations),
                'total_records': len(dataset_records),
                'structure': 'Each record contains id (string) and conversations array with from/value objects'
            }
        
        # Save main dataset as array (not wrapped in object)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dataset_records, f, indent=2, ensure_ascii=False)
        
        # Save metadata separately with format information
        metadata_path = output_path.with_suffix('.metadata.json')
        metadata = {
            'generation_stats': asdict(self.stats),
            'configuration': {
                'chunk_size': config.chunk_size,
                'chunk_overlap': config.chunk_overlap,
                'max_conversations_per_chunk': config.max_conversations_per_chunk,
                'temperature': config.temperature,
                'max_tokens': config.max_tokens,
                'dataset_format': config.dataset_format
            },
            'conversation_metadata': [entry.metadata for entry in conversations],
            'format_info': format_info
        }
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(conversations)} conversation pairs ({len(dataset_records)} total records) to {output_path}")
        logger.info(f"Dataset format: {config.dataset_format}")
        logger.info(f"Saved metadata to {metadata_path}")
        
        # Upload to Hugging Face if enabled
        if self.hf_uploader and config.enable_hf_upload:
            logger.info("Uploading dataset to Hugging Face Hub...")
            try:
                upload_success = self.hf_uploader.upload_dataset(output_path, hf_repo_name)
                if upload_success:
                    logger.info("Successfully uploaded dataset to Hugging Face Hub!")
                else:
                    logger.warning("Failed to upload dataset to Hugging Face Hub")
            except Exception as e:
                logger.error(f"Error uploading to Hugging Face Hub: {e}")
        
        return output_path
    
    def _log_generation_stats(self):
        logger.info("--- Generation Statistics ---")
        logger.info(f"Total PDFs Processed: {self.stats.total_pdfs_processed}")
        logger.info(f"Total Chunks Extracted: {self.stats.total_chunks_extracted}")
        successful_chunk_generations = self.stats.successful_generations 
        logger.info(f"Chunks with Successful Conversation Generations: {successful_chunk_generations}")
        logger.info(f"Chunks with Failed/Empty Conversation Generations: {self.stats.failed_generations}")
        logger.info(f"Total Conversation Pairs Generated: {self.stats.total_conversations_generated}")
        
        if self.stats.total_chunks_extracted > 0 :
            success_rate = (successful_chunk_generations / self.stats.total_chunks_extracted) * 100
            logger.info(f"Chunk-level Generation Success Rate: {success_rate:.2f}%")

        avg_score = self.stats.average_confidence_score if self.stats.total_conversations_generated > 0 else 0.0
        logger.info(f"Average Quality Score (of generated pairs): {avg_score:.2f}")
        logger.info(f"Processing Time: {self.stats.processing_time_seconds:.2f} seconds")
        logger.info(f"Models Used: {', '.join(self.stats.models_used) if self.stats.models_used else 'N/A'}")
        logger.info("-----------------------------")
    
    async def generate_sample_dataset(self, 
                                      num_samples: int = 50, 
                                      hf_repo_name: Optional[str] = None,
                                      progress_bar: Optional[Progress] = None,
                                      main_task_id: Optional[Any] = None
                                      ) -> Path:
        """Generate a small sample dataset (fixed number of conversations)."""
        # This will now also use the progress bar if provided
        return await self.generate_dataset(
            max_conversations=num_samples,
            hf_repo_name=hf_repo_name,
            progress_bar=progress_bar,
            main_task_id=main_task_id
        )
    
    def get_statistics(self) -> GenerationStats:
        """Get current generation statistics."""
        return self.stats