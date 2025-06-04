"""
Processes multiple PDF files in parallel to generate a unified dataset of conversation pairs,
then uploads it to Hugging Face.
"""
import asyncio
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import asdict, dataclass # Ensure dataclass is imported if DatasetEntry is defined here
from datetime import datetime

from loguru import logger

# Assuming these modules and classes exist in the specified locations
from config import config # Global configuration
from src.pdf_processor import PDFProcessor, TextChunk
from src.ai_models import AIModelManager, ConversationPair
from src.huggingface_uploader import HuggingFaceUploader
# Re-defining DatasetEntry here for clarity or importing if it's globally accessible and meant to be.
# For now, let's assume it's defined in dataset_generator and we can import it.
from src.dataset_generator import DatasetEntry # Preferred

# --- Quality Filter (adapted from MedicalDatasetGenerator) ---
# This might need access to the config object if it uses config.min_conversation_length etc.
def _passes_quality_filters(conversation: ConversationPair, min_length: int, max_length: int) -> bool:
    """
    Basic quality filters for generated conversations.
    Adapted from MedicalDatasetGenerator._passes_quality_filters.
    """
    if not conversation.user_message or not conversation.assistant_message:
        logger.warning("Skipping conversation: Empty user or assistant message.")
        return False

    user_len = len(conversation.user_message)
    asst_len = len(conversation.assistant_message)

    if not (min_length <= user_len <= max_length and min_length <= asst_len <= max_length):
        logger.warning(
            f"Skipping conversation: Length constraints not met. "
            f"User: {user_len}, Assistant: {asst_len} (Min: {min_length}, Max: {max_length})"
        )
        return False
    
    # Add more sophisticated checks if needed (e.g., based on confidence_score)
    if conversation.confidence_score < 0.5: # Example threshold
        logger.warning(f"Skipping conversation: Low confidence score ({conversation.confidence_score:.2f})")
        return False
        
    # Placeholder for profanity or sensitive content filter
    # if contains_profanity(conversation.user_message) or contains_profanity(conversation.assistant_message):
    #     logger.warning("Skipping conversation: Contains potentially inappropriate content.")
    #     return False
        
    return True

# --- Worker function to process a single PDF ---
async def process_single_pdf_to_entries(
    pdf_path: Path, 
    pdf_processor_instance: PDFProcessor, 
    ai_manager_instance: AIModelManager
) -> List[DatasetEntry]:
    """
    Processes a single PDF: extracts chunks, generates conversations for each chunk,
    filters them, and converts them to DatasetEntry objects.
    """
    logger.info(f"Starting processing for PDF: {pdf_path.name}")
    pdf_dataset_entries: List[DatasetEntry] = []

    try:
        text_chunks: List[TextChunk] = pdf_processor_instance.extract_text_from_pdf(pdf_path)
    except Exception as e:
        logger.error(f"Error extracting text chunks from {pdf_path.name}: {e}")
        return [] # Return empty list if PDF processing fails at chunking stage

    if not text_chunks:
        logger.warning(f"No text chunks extracted from {pdf_path.name}")
        return []

    logger.info(f"Extracted {len(text_chunks)} chunks from {pdf_path.name}")

    for i, chunk in enumerate(text_chunks):
        logger.debug(f"Processing chunk {i+1}/{len(text_chunks)} from {pdf_path.name}")
        try:
            conversation_pairs: List[ConversationPair] = await ai_manager_instance.generate_conversations(
                chunk.content,
                config.max_conversations_per_chunk
            )

            if conversation_pairs:
                processed_count = 0
                for conv_pair in conversation_pairs:
                    # Apply quality filters
                    if not _passes_quality_filters(conv_pair, config.min_conversation_length, config.max_conversation_length):
                        continue

                    entry = DatasetEntry(
                        user_message=conv_pair.user_message,
                        assistant_message=conv_pair.assistant_message,
                        metadata={
                            'source_pdf': pdf_path.name,
                            'page_number': chunk.page_number,
                            'chunk_index': chunk.chunk_index, # Ensure TextChunk has chunk_index
                            'extraction_method': chunk.extraction_method, # Ensure TextChunk has extraction_method
                            'confidence_score': conv_pair.confidence_score,
                            'medical_context': conv_pair.medical_context,
                            'generation_metadata': conv_pair.generation_metadata,
                            'timestamp': datetime.now().isoformat(),
                            'user_message_length': len(conv_pair.user_message),
                            'assistant_message_length': len(conv_pair.assistant_message)
                        }
                    )
                    pdf_dataset_entries.append(entry)
                    processed_count +=1
                logger.debug(f"Generated and processed {processed_count} entries for chunk {chunk.chunk_index} from {pdf_path.name}")
            else:
                logger.debug(f"No conversation pairs generated for chunk {chunk.chunk_index} from {pdf_path.name}")

        except Exception as e:
            logger.error(f"Error generating/processing conversations for chunk {chunk.chunk_index} from {pdf_path.name}: {e}")
            # Continue to the next chunk
        
        # Add a small delay to be polite to the API, especially when many PDFs are processed in parallel
        await asyncio.sleep(config.api_call_delay_seconds) # Use the new config attribute

    logger.info(f"Finished processing PDF: {pdf_path.name}. Generated {len(pdf_dataset_entries)} entries.")
    return pdf_dataset_entries

# --- Main parallel processing function ---
async def main_parallel_processing():
    """
    Main orchestrator for parallel PDF processing, dataset aggregation, saving, and uploading.
    """
    logger.info("Starting parallel PDF processing...")
    start_time = datetime.now()

    # Initialize components
    pdf_processor = PDFProcessor()
    ai_manager = AIModelManager() # This might load models, could be slow
    
    # Set stop event for AI manager if it supports it (as in MedicalDatasetGenerator)
    # stop_event = asyncio.Event() # If we need to handle graceful shutdown
    # if hasattr(ai_manager, 'set_stop_event'):
    #     ai_manager.set_stop_event(stop_event)

    hf_uploader: Optional[HuggingFaceUploader] = None
    if config.enable_hf_upload:
        logger.info("Hugging Face upload is enabled. Initializing uploader...")
        hf_uploader = HuggingFaceUploader()
        if not hf_uploader.authenticate(): # Authenticate early
            logger.error("Hugging Face authentication failed. Upload will be skipped.")
            hf_uploader = None # Disable upload if auth fails
    else:
        logger.info("Hugging Face upload is disabled.")

    # List PDF files
    input_dir = config.input_dir
    pdf_files_str = list(input_dir.glob("*.pdf"))
    pdf_files_str.extend(list(input_dir.glob("*.PDF"))) # Case-insensitive
    pdf_files = list(set(pdf_files_str)) # Remove duplicates

    if not pdf_files:
        logger.warning(f"No PDF files found in {input_dir}. Exiting.")
        return

    logger.info(f"Found {len(pdf_files)} PDF(s) to process: {[f.name for f in pdf_files]}")

    # Create and run tasks in parallel
    # Adjust max_concurrent_tasks as needed, e.g., based on API rate limits or CPU cores
    # For I/O bound (like API calls) and CPU-bound (like PDF parsing) mix,
    # a semaphore can be useful to limit true concurrency for CPU parts if needed,
    # but asyncio.gather handles concurrent I/O calls well.
    # Let's not overcomplicate with semaphores unless rate limits become an issue.
    
    tasks = [
        process_single_pdf_to_entries(pdf_file, pdf_processor, ai_manager)
        for pdf_file in pdf_files
    ]
    
    logger.info(f"--- Launching {len(tasks)} parallel PDF processing tasks for the above files ---")
    # return_exceptions=True allows us to get results even if some tasks fail
    results_from_tasks: List[List[DatasetEntry] | Exception] = await asyncio.gather(*tasks, return_exceptions=True)

    all_dataset_entries: List[DatasetEntry] = []
    successful_pdf_details: List[str] = []
    failed_pdf_details: List[str] = []
    
    pdfs_processed_count = 0
    total_chunks_from_successful_pdfs = 0 # Track chunks only from successfully processed PDFs for entry count
    # For ETA, we might need total expected chunks. Let's start with PDF-based progress.

    logger.info("--- Individual PDF Processing Results ---") # Header for this section
    initial_start_time_for_eta = datetime.now() # For overall ETA calculation

    for i, result in enumerate(results_from_tasks):
        pdfs_processed_count += 1
        current_time = datetime.now()
        time_elapsed_seconds = (current_time - initial_start_time_for_eta).total_seconds()

        pdf_file_path = pdf_files[i] # Get the Path object
        pdf_name = pdf_file_path.name
        
        num_entries_this_pdf = 0
        processed_successfully_this_pdf = False

        if isinstance(result, Exception):
            logger.error(f"[FAILURE] Processing PDF '{pdf_name}': {type(result).__name__} - {result}")
            failed_pdf_details.append(f"{pdf_name} (Error: {type(result).__name__} - {str(result).splitlines()[0] if str(result) else 'Unknown error'})")
        elif isinstance(result, list): # Expected result is List[DatasetEntry]
            num_entries_this_pdf = len(result)
            logger.info(f"[SUCCESS] Processing PDF '{pdf_name}': Completed, {num_entries_this_pdf} entries generated.")
            all_dataset_entries.extend(result)
            successful_pdf_details.append(f"{pdf_name} ({num_entries_this_pdf} entries)")
            # To calculate chunk-based speed, we need num_chunks for this PDF.
            # This info is inside the 'result' if we modify process_single_pdf_to_entries to return it.
            # For now, let's focus on PDF-based progress.
            processed_successfully_this_pdf = True 
            # If process_single_pdf_to_entries returned (entries_list, num_chunks_in_pdf):
            # total_chunks_from_successful_pdfs += num_chunks_in_pdf 
        else:
            logger.error(f"[UNKNOWN] Unexpected result type for PDF '{pdf_name}': {type(result)}")
            failed_pdf_details.append(f"{pdf_name} (Unknown result type: {type(result).__name__})")

        # Log Progress
        pdfs_remaining = len(pdf_files) - pdfs_processed_count
        if time_elapsed_seconds > 0:
            speed_pdfs_per_sec = pdfs_processed_count / time_elapsed_seconds
            if speed_pdfs_per_sec > 0:
                eta_seconds = pdfs_remaining / speed_pdfs_per_sec
                eta_str = f"{eta_seconds // 3600:02.0f}h {(eta_seconds % 3600) // 60:02.0f}m {eta_seconds % 60:02.0f}s"
                logger.info(f"Progress: {pdfs_processed_count}/{len(pdf_files)} PDFs processed. Speed: {speed_pdfs_per_sec:.2f} PDFs/sec. ETA: {eta_str}")
            else:
                logger.info(f"Progress: {pdfs_processed_count}/{len(pdf_files)} PDFs processed. (Calculating ETA...)")
        else:
            logger.info(f"Progress: {pdfs_processed_count}/{len(pdf_files)} PDFs processed.")

    successful_pdfs_count = len(successful_pdf_details)
    failed_pdfs_count = len(failed_pdf_details)

    logger.info("--- Overall Processing Summary ---")
    logger.info(f"Total PDFs initially found: {len(pdf_files)}")
    
    logger.info(f"Successfully processed PDFs ({successful_pdfs_count}):")
    if successful_pdf_details:
        for detail in successful_pdf_details:
            logger.info(f"  + {detail}")
    else:
        logger.info("  (None successfully processed)")
    
    logger.info(f"Failed to process PDFs ({failed_pdfs_count}):")
    if failed_pdf_details:
        for detail in failed_pdf_details:
            logger.warning(f"  - {detail}") # Use warning for failed files
    else:
        logger.info("  (None failed)")

    logger.info(f"Total dataset entries aggregated from all successful PDFs: {len(all_dataset_entries)}")

    if not all_dataset_entries:
        logger.warning("No dataset entries were generated from any PDF. Exiting.")
        processing_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Total processing time: {processing_time:.2f} seconds.")
        return

    # Save the unified dataset
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename_base = f"unified_dataset_{timestamp}"
    output_jsonl_filename = f"{output_filename_base}.jsonl"
    output_jsonl_path = config.output_dir / output_jsonl_filename
    
    # Ensure output directory exists (config should handle this, but double check)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving unified dataset to {output_jsonl_path}...")
    try:
        with open(output_jsonl_path, 'w', encoding='utf-8') as f:
            for entry in all_dataset_entries:
                f.write(json.dumps(asdict(entry)) + '\n')
        logger.info(f"Unified dataset saved successfully to {output_jsonl_path}.")
    except IOError as e:
        logger.error(f"Failed to save unified dataset: {e}")
        # Decide if we should exit or attempt upload if enabled
        if not config.enable_hf_upload:
            return # Exit if save fails and upload is not enabled.

    # TODO: Optionally generate and save a .metadata.json file similar to MedicalDatasetGenerator._save_dataset
    # For now, we'll skip the detailed metadata file generation to simplify.
    # The uploader will generate a README.md

    # Upload to Hugging Face if enabled and saving was successful (or if we proceed despite save error)
    if config.enable_hf_upload and hf_uploader:
        if output_jsonl_path.exists(): # Only upload if file was actually created
            logger.info(f"Attempting to upload {output_jsonl_path.name} to Hugging Face...")
            
            # Determine repo name: Use config.hf_repo_name if set, otherwise uploader will auto-generate
            repo_name_to_upload = config.hf_repo_name
            if not repo_name_to_upload:
                 # Let uploader handle auto-generation based on filename, or set a default project one
                 # Example: repo_name_to_upload = f"user/{output_filename_base}" - needs HF username
                 # The uploader has logic to create from filename if repo_name is None
                 logger.info("hf_repo_name not set in config, HuggingFaceUploader will attempt to auto-generate it.")


            success = hf_uploader.upload_dataset(
                dataset_path=output_jsonl_path,
                repo_name=repo_name_to_upload, # Can be None for auto-generation by uploader
                private=config.hf_dataset_private,
                commit_message=f"{config.hf_commit_message} - Parallel Batch {timestamp}"
            )
            if success:
                logger.info(f"Successfully uploaded dataset to Hugging Face.")
            else:
                logger.error(f"Failed to upload dataset to Hugging Face.")
        else:
            logger.error(f"Output file {output_jsonl_path} not found. Skipping Hugging Face upload.")
    
    processing_time = (datetime.now() - start_time).total_seconds()
    logger.info(f"Total processing time: {processing_time:.2f} seconds.")
    logger.info("Parallel PDF processing script finished.")


if __name__ == "__main__":
    # Configure Loguru to write to a file
    log_file_path = Path("logs") / "parallel_processing.log"
    log_file_path.parent.mkdir(parents=True, exist_ok=True) # Ensure logs directory exists
    logger.add(log_file_path, rotation="10 MB", retention="7 days", level="DEBUG") # Log DEBUG and above to file
    
    # Standard console output (default is INFO)
    # logger.add(sys.stderr, level="INFO") # Already added by default

    try:
        asyncio.run(main_parallel_processing())
    except KeyboardInterrupt:
        logger.warning("Process interrupted by user (KeyboardInterrupt).")
    except Exception as e:
        logger.opt(exception=True).critical(f"An unhandled error occurred in main_parallel_processing: {e}") 