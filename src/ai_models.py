"""AI Models module for generating medical conversations using DeepSeek, OpenAI-compatible APIs, and local Hugging Face models."""

import json
import asyncio
import torch
import httpx
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import re

from openai import AsyncOpenAI
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    AutoModelForSeq2SeqLM,
    pipeline,
    BitsAndBytesConfig
)
from loguru import logger

from config import config


class ModelProvider(Enum):
    """Supported AI model providers."""
    DEEPSEEK = "deepseek"
    OPENAI = "openai"
    HUGGINGFACE_LOCAL = "huggingface_local"


@dataclass
class ConversationPair:
    """Represents a user-assistant conversation pair."""
    user_message: str
    assistant_message: str
    medical_context: str
    confidence_score: float
    generation_metadata: Dict[str, Any]


class BaseAIModel(ABC):
    """Abstract base class for AI models."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.stop_event: Optional[asyncio.Event] = None
    
    def set_stop_event(self, stop_event: asyncio.Event):
        """Set the event used to signal a stop request."""
        self.stop_event = stop_event
    
    @abstractmethod
    async def generate_conversations(self, medical_text: str, num_conversations: int) -> List[ConversationPair]:
        """Generate conversation pairs from medical text."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the model is available and properly configured."""
        pass


class DeepSeekModel(BaseAIModel):
    """DeepSeek API model implementation."""
    
    def __init__(self, model_name: str = "deepseek-chat"):
        super().__init__(model_name)
        if not config.deepseek_api_key:
            raise ValueError("DeepSeek API key not configured")
        
        self.api_key = config.deepseek_api_key
        self.base_url = "https://api.deepseek.com"
        
    def is_available(self) -> bool:
        """Check if DeepSeek API is available."""
        try:
            # Test with a minimal request
            async def test_request():
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{self.base_url}/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json"
                        },
                        json={
                            "model": self.model_name,
                            "messages": [{"role": "user", "content": "Hello"}],
                            "max_tokens": 10
                        },
                        timeout=60.0
                    )
                    return response.status_code == 200
            
            # Check if we're already in an event loop
            import asyncio
            try:
                loop = asyncio.get_running_loop()
                # We're in an event loop, create a task and run it
                task = asyncio.create_task(test_request())
                # We need to wait for it somehow - let's use asyncio.wait_for with a timeout
                import concurrent.futures
                import threading
                
                # Create a new event loop in a separate thread
                def run_in_new_loop():
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        return new_loop.run_until_complete(test_request())
                    finally:
                        new_loop.close()
                
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(run_in_new_loop)
                    return future.result(timeout=15)
                    
            except RuntimeError:
                # No event loop running, safe to use asyncio.run()
                return asyncio.run(test_request())
            
        except Exception as e:
            logger.warning(f"DeepSeek API not available: {e}")
            return False
    
    async def generate_conversations(self, medical_text: str, num_conversations: int) -> List[ConversationPair]:
        """Generate conversation pairs using DeepSeek API."""
        if self.stop_event and self.stop_event.is_set():
            logger.info("Stop requested before DeepSeek API call.")
            return []
        
        prompt = self._get_generation_prompt(medical_text, num_conversations)
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": self.model_name,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": config.temperature,
                        "max_tokens": config.max_tokens,
                        "response_format": {"type": "json_object"}
                    },
                    timeout=60.0
                )
                
                if response.status_code != 200:
                    logger.error(f"DeepSeek API error: {response.status_code} - {response.text}")
                    return []
                
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                
                # Parse JSON response
                try:
                    conversations_data = json.loads(content)
                except json.JSONDecodeError:
                    logger.warning("Failed to parse JSON from DeepSeek, attempting text extraction")
                    conversations_data = self._extract_conversations_from_text(content, num_conversations)
                
                # Track usage
                usage_info = result.get("usage", {})
                
                return self._parse_conversations(
                    conversations_data, 
                    medical_text[:200], 
                    usage_info
                )
                
        except httpx.ReadTimeout:
            logger.warning(f"DeepSeek API call timed out for model {self.model_name}")
            return []
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from DeepSeek API: {e}. Response: {content}")
            return []
        except Exception as e:
            if self.stop_event and self.stop_event.is_set():
                logger.info(f"DeepSeek API call potentially interrupted by stop request: {e}")
                return []
            logger.error(f"Error calling DeepSeek API: {e}")
            return []
    
    def _get_generation_prompt(self, medical_text: str, num_conversations: int) -> str:
        """Generate the prompt for DeepSeek."""
        return f"""You are a medical AI assistant creating educational conversation examples for training purposes.

Create {num_conversations} realistic but educational conversation examples between patients and medical AI assistants based on this medical reference text:

MEDICAL REFERENCE TEXT:
{medical_text[:1500]}

Create diverse, educational conversations that demonstrate:
- Common health questions and informational responses
- General wellness discussions
- Questions about medical concepts from the reference text
- Educational health information requests

IMPORTANT GUIDELINES:
- This is for educational/training purposes only
- Include clear medical disclaimers in responses
- Focus on general health information, not specific diagnoses
- Responses should encourage consulting healthcare professionals
- Keep conversations informational and educational
- Avoid specific medical advice or treatment recommendations

Format as valid JSON with this structure:
{{
  "conversations": [
    {{
      "user": "educational health question",
      "assistant": "informational response with disclaimers", 
      "context": "educational context"
    }}
  ]
}}

JSON Response:"""
    
    def _extract_conversations_from_text(self, content: str, num_conversations: int) -> Dict:
        """Extract conversations from unstructured text if JSON parsing fails."""
        conversations = []
        
        # Simple pattern matching for fallback
        lines = content.split('\n')
        current_conv = {}
        
        for line in lines:
            line = line.strip()
            if line.startswith(('User:', 'Patient:', 'Q:')):
                if current_conv and 'user' in current_conv and 'assistant' in current_conv:
                    conversations.append(current_conv)
                current_conv = {'user': line.split(':', 1)[1].strip()}
            elif line.startswith(('Assistant:', 'AI:', 'A:')):
                if 'user' in current_conv:
                    current_conv['assistant'] = line.split(':', 1)[1].strip()
                    current_conv['context'] = "Generated from text"
        
        # Add the last conversation
        if current_conv and 'user' in current_conv and 'assistant' in current_conv:
            conversations.append(current_conv)
        
        return {"conversations": conversations[:num_conversations]}
    
    def _parse_conversations(self, result: Dict, medical_context: str, usage_info: Dict) -> List[ConversationPair]:
        """Parse DeepSeek's response into ConversationPair objects."""
        conversations = []
        
        if "conversations" not in result:
            logger.warning("Invalid response format from DeepSeek")
            return conversations
        
        for i, conv in enumerate(result["conversations"]):
            if "user" in conv and "assistant" in conv:
                conversations.append(ConversationPair(
                    user_message=conv["user"],
                    assistant_message=conv["assistant"],
                    medical_context=conv.get("context", medical_context[:200]),
                    confidence_score=self._calculate_quality_score(conv),
                    generation_metadata={
                        "model": self.model_name,
                        "provider": "deepseek",
                        "usage": usage_info,
                        "conversation_index": i
                    }
                ))
        
        return conversations
    
    def _calculate_quality_score(self, conversation: Dict) -> float:
        """Calculate a quality score for the generated conversation."""
        user_msg = conversation.get("user", "")
        assistant_msg = conversation.get("assistant", "")
        
        score = 0.5
        
        if 20 <= len(user_msg) <= 300:
            score += 0.2
        if 50 <= len(assistant_msg) <= 500:
            score += 0.2
        
        medical_terms = ["symptom", "doctor", "pain", "treatment", "medication", "condition", "health"]
        if any(term in user_msg.lower() for term in medical_terms):
            score += 0.1
        
        return min(1.0, score)


class OpenAICompatibleModel(BaseAIModel):
    """OpenAI-compatible API model implementation (supports OpenRouter, OpenAI, etc.)."""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        super().__init__(model_name)
        if not config.openai_api_key:
            raise ValueError("OpenAI API key not configured")
        
        # Configure client for OpenAI-compatible APIs
        client_config = {
            "api_key": config.openai_api_key,
        }
        
        if config.openai_base_url:
            client_config["base_url"] = config.openai_base_url
        
        if config.openai_organization:
            client_config["organization"] = config.openai_organization
            
        # Add explicit timeouts
        # Connect timeout: time to establish connection
        # Read timeout: time to wait for server to send response after connection
        # These can be made configurable in config.py if needed later
        client_config["timeout"] = httpx.Timeout(60.0, read=120.0) # 60s connect, 120s read
            
        self.client = AsyncOpenAI(**client_config)
        
    def is_available(self) -> bool:
        """Check if OpenAI-compatible API is available."""
        try:
            # Test with a minimal request
            async def test_request():
                try:
                    response = await self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[{"role": "user", "content": "Hello"}],
                        max_tokens=10
                    )
                    return True
                except Exception:
                    return False
            
            # Check if we're already in an event loop
            import asyncio
            try:
                loop = asyncio.get_running_loop()
                # We're in an event loop, create a new event loop in a separate thread
                import concurrent.futures
                
                # Create a new event loop in a separate thread
                def run_in_new_loop():
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        return new_loop.run_until_complete(test_request())
                    finally:
                        new_loop.close()
                
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(run_in_new_loop)
                    return future.result(timeout=15)
                    
            except RuntimeError:
                # No event loop running, safe to use asyncio.run()
                return asyncio.run(test_request())
            
        except Exception as e:
            logger.warning(f"OpenAI-compatible API not available: {e}")
            return False
    
    async def generate_conversations(self, medical_text: str, num_conversations: int) -> List[ConversationPair]:
        """Generate conversations using OpenAI-compatible API."""
        if self.stop_event and self.stop_event.is_set():
            logger.info(f"Stop requested before {self.model_name} API call.")
            return []
        
        try:
            prompt = self._get_generation_prompt(medical_text, num_conversations)
            
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                response_format={"type": "json_object"} if "gpt" in self.model_name.lower() else None
            )
            
            content = response.choices[0].message.content
            logger.debug(f"Raw response content: {content[:200]}...")
            
            # Enhanced JSON parsing with multiple fallback strategies
            conversations_data = self._parse_response_content(content, num_conversations)
            
            # Track usage
            usage_info = {
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0
            }
            
            return self._parse_conversations(
                conversations_data, 
                medical_text[:200], 
                usage_info
            )
            
        except httpx.ReadTimeout:
            logger.warning(f"{self.model_name} API call timed out.")
            return []

    def _parse_response_content(self, content: str, num_conversations: int) -> Dict:
        """Enhanced response parsing with multiple fallback strategies."""
        if not content:
            logger.warning("Empty response content received")
            return {"conversations": []}
        
        # Strategy 1: Direct JSON parsing
        try:
            conversations_data = json.loads(content)
            if "conversations" in conversations_data:
                logger.debug("Successfully parsed JSON response")
                return conversations_data
        except json.JSONDecodeError as e:
            logger.debug(f"Direct JSON parsing failed: {e}")
        
        # Strategy 2: Extract JSON from mixed content
        try:
            # Look for JSON block in the response
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                conversations_data = json.loads(json_str)
                if "conversations" in conversations_data:
                    logger.debug("Successfully extracted JSON from mixed content")
                    return conversations_data
        except (json.JSONDecodeError, AttributeError) as e:
            logger.debug(f"JSON extraction from mixed content failed: {e}")
        
        # Strategy 3: Enhanced text extraction
        logger.warning("Failed to parse JSON from OpenAI-compatible API, attempting enhanced text extraction")
        return self._extract_conversations_from_text(content, num_conversations)

    def _extract_conversations_from_text(self, content: str, num_conversations: int) -> Dict:
        """Enhanced conversation extraction from unstructured text."""
        conversations = []
        
        # Clean the content
        content = content.strip()
        lines = content.split('\n')
        
        current_conv = {}
        
        # Enhanced patterns for different response formats
        user_patterns = [
            r'^(?:User|Patient|Question|Q):\s*(.+)$',
            r'^(?:\d+\.?\s*)?(?:User|Patient|Question|Q):\s*(.+)$',
            r'(?i)(?:user|patient|question)\s*:\s*(.+)$',
        ]
        
        assistant_patterns = [
            r'^(?:Assistant|AI|Answer|A|Response):\s*(.+)$',
            r'^(?:\d+\.?\s*)?(?:Assistant|AI|Answer|A|Response):\s*(.+)$',
            r'(?i)(?:assistant|ai|answer|response)\s*:\s*(.+)$',
        ]
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check for user message patterns
            user_match = None
            for pattern in user_patterns:
                user_match = re.match(pattern, line, re.IGNORECASE)
                if user_match:
                    break
            
            if user_match:
                # Save previous conversation if complete
                if current_conv and 'user' in current_conv and 'assistant' in current_conv:
                    conversations.append(current_conv)
                    if len(conversations) >= num_conversations:
                        break
                
                current_conv = {'user': user_match.group(1).strip()}
                continue
            
            # Check for assistant message patterns
            assistant_match = None
            for pattern in assistant_patterns:
                assistant_match = re.match(pattern, line, re.IGNORECASE)
                if assistant_match:
                    break
            
            if assistant_match and 'user' in current_conv:
                current_conv['assistant'] = assistant_match.group(1).strip()
                current_conv['context'] = "Generated from text extraction"
                continue
            
            # If we have a user message but no assistant match, this might be continuation
            if 'user' in current_conv and 'assistant' not in current_conv and line:
                # Check if this looks like an assistant response (no label)
                if not any(re.match(pattern, line, re.IGNORECASE) for pattern in user_patterns):
                    current_conv['assistant'] = line
                    current_conv['context'] = "Generated from text extraction"
        
        # Add the last conversation if complete
        if current_conv and 'user' in current_conv and 'assistant' in current_conv:
            conversations.append(current_conv)
        
        # If we still don't have conversations, try a more aggressive approach
        if not conversations:
            conversations = self._fallback_text_extraction(content, num_conversations)
        
        logger.info(f"Extracted {len(conversations)} conversations from text")
        return {"conversations": conversations[:num_conversations]}

    def _fallback_text_extraction(self, content: str, num_conversations: int) -> List[Dict]:
        """Fallback method for extracting conversations when standard patterns fail."""
        conversations = []
        
        # Split content into potential conversation blocks
        blocks = re.split(r'\n\s*\n', content)
        
        for block in blocks[:num_conversations * 2]:  # Look at more blocks
            block = block.strip()
            if len(block) < 20:  # Skip very short blocks
                continue
            
            # Try to split the block into question and answer
            sentences = re.split(r'[.!?]+', block)
            if len(sentences) >= 2:
                # Take the first sentence as user, rest as assistant
                user_msg = sentences[0].strip()
                assistant_msg = '. '.join(sentences[1:]).strip()
                
                if len(user_msg) > 10 and len(assistant_msg) > 20:
                    conversations.append({
                        'user': user_msg,
                        'assistant': assistant_msg,
                        'context': 'Fallback extraction'
                    })
                    
                    if len(conversations) >= num_conversations:
                        break
        
        return conversations
    
    def _get_generation_prompt(self, medical_text: str, num_conversations: int) -> str:
        """Generate the prompt for OpenAI-compatible APIs."""
        return f"""You are a medical AI assistant creating educational conversation examples for training purposes.

Create exactly {num_conversations} realistic but educational conversation examples between patients and medical AI assistants based on this medical reference text:

MEDICAL REFERENCE TEXT:
{medical_text[:1500]}

Create diverse, educational conversations that demonstrate:
- Common health questions and informational responses
- General wellness discussions
- Questions about medical concepts from the reference text
- Educational health information requests

IMPORTANT GUIDELINES:
- This is for educational/training purposes only
- Include clear medical disclaimers in responses
- Focus on general health information, not specific diagnoses
- Responses should encourage consulting healthcare professionals
- Keep conversations informational and educational
- Avoid specific medical advice or treatment recommendations

CRITICAL: You must respond with ONLY valid JSON in exactly this format (no additional text, explanations, or formatting):

{{
  "conversations": [
    {{
      "user": "What are the common symptoms of diabetes?",
      "assistant": "Common symptoms of diabetes include increased thirst, frequent urination, unexplained weight loss, fatigue, and blurred vision. However, symptoms can vary, and some people may not experience obvious symptoms initially. It's important to consult with a healthcare professional for proper screening and diagnosis if you have concerns about diabetes.",
      "context": "educational health information"
    }},
    {{
      "user": "How can I prevent heart disease?",
      "assistant": "Heart disease prevention typically involves maintaining a healthy lifestyle including regular exercise, a balanced diet low in saturated fats, not smoking, limiting alcohol, and managing stress. Regular check-ups with your healthcare provider are also important for monitoring risk factors like blood pressure and cholesterol. Please consult with your doctor for personalized prevention strategies.",
      "context": "preventive health education"
    }}
  ]
}}

Generate exactly {num_conversations} conversations in this exact JSON format:"""
    
    def _parse_conversations(self, result: Dict, medical_context: str, usage_info: Dict) -> List[ConversationPair]:
        """Parse OpenAI-compatible API response into ConversationPair objects."""
        conversations = []
        
        if not isinstance(result, dict):
            logger.warning(f"Invalid response format from OpenAI-compatible API - not a dict: {type(result)}")
            return conversations
        
        if "conversations" not in result:
            logger.warning(f"Invalid response format from OpenAI-compatible API - missing 'conversations' key. Keys found: {list(result.keys())}")
            return conversations
        
        conversations_list = result["conversations"]
        if not isinstance(conversations_list, list):
            logger.warning(f"Invalid conversations format - not a list: {type(conversations_list)}")
            return conversations
        
        logger.info(f"Processing {len(conversations_list)} conversations from API response")
        
        for i, conv in enumerate(conversations_list):
            if not isinstance(conv, dict):
                logger.warning(f"Conversation {i} is not a dict: {type(conv)}")
                continue
                
            if "user" not in conv or "assistant" not in conv:
                logger.warning(f"Conversation {i} missing required keys. Keys found: {list(conv.keys())}")
                continue
            
            user_msg = conv["user"]
            assistant_msg = conv["assistant"]
            
            if not user_msg or not assistant_msg:
                logger.warning(f"Conversation {i} has empty messages - user: {bool(user_msg)}, assistant: {bool(assistant_msg)}")
                continue
            
            conversations.append(ConversationPair(
                user_message=user_msg,
                assistant_message=assistant_msg,
                medical_context=conv.get("context", medical_context[:200]),
                confidence_score=self._calculate_quality_score(conv),
                generation_metadata={
                    "model": self.model_name,
                    "provider": "openai_compatible",
                    "usage": usage_info,
                    "conversation_index": i
                }
            ))
        
        logger.info(f"Successfully parsed {len(conversations)} valid conversations")
        return conversations
    
    def _calculate_quality_score(self, conversation: Dict) -> float:
        """Calculate a quality score for the generated conversation."""
        user_msg = conversation.get("user", "")
        assistant_msg = conversation.get("assistant", "")
        
        score = 0.5
        
        if 20 <= len(user_msg) <= 300:
            score += 0.2
        if 50 <= len(assistant_msg) <= 500:
            score += 0.2
        
        medical_terms = ["symptom", "doctor", "pain", "treatment", "medication", "condition", "health"]
        if any(term in user_msg.lower() for term in medical_terms):
            score += 0.1
        
        return min(1.0, score)


class HuggingFaceLocalModel(BaseAIModel):
    """Local Hugging Face model implementation."""
    
    def __init__(self, model_name: str = None):
        model_name = model_name or config.local_model_name
        super().__init__(model_name)
        
        self.tokenizer = None
        self.model = None
        self.device = self._get_device()
        self.pipeline = None
        
        # Initialize model
        self._load_model()
    
    def _get_device(self) -> str:
        """Determine the best device to use."""
        if config.device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return config.device
    
    def _load_model(self):
        """Load the Hugging Face model and tokenizer."""
        try:
            logger.info(f"Loading model {self.model_name} on device {self.device}")
            
            # Configure quantization if enabled
            quantization_config = None
            if config.load_in_8bit and self.device == "cuda":
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0,
                )
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=config.models_cache_dir,
                trust_remote_code=True
            )
            
            # Add pad token if it doesn't exist
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Try to load as a text generation model first
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    cache_dir=config.models_cache_dir,
                    quantization_config=quantization_config,
                    device_map="auto" if self.device == "cuda" else None,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    trust_remote_code=True
                )
                
                # Create text generation pipeline
                self.pipeline = pipeline(
                    "text-generation",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device=0 if self.device == "cuda" else -1,
                    return_full_text=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                
            except Exception:
                # Fallback to seq2seq model
                logger.info("Trying as sequence-to-sequence model")
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.model_name,
                    cache_dir=config.models_cache_dir,
                    quantization_config=quantization_config,
                    device_map="auto" if self.device == "cuda" else None,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    trust_remote_code=True
                )
                
                # Create text2text generation pipeline
                self.pipeline = pipeline(
                    "text2text-generation",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device=0 if self.device == "cuda" else -1,
                    return_full_text=False
                )
            
            logger.info(f"Successfully loaded {self.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            raise
    
    def is_available(self) -> bool:
        """Check if local model is loaded and available."""
        return self.model is not None and self.pipeline is not None
    
    async def generate_conversations(self, medical_text: str, num_conversations: int) -> List[ConversationPair]:
        """Generate conversation pairs using local Hugging Face model."""
        if not self.is_available():
            logger.warning(f"Local model {self.model_name} not available. Skipping generation.")
            return []

        all_pairs = []
        for i in range(num_conversations):
            if self.stop_event and self.stop_event.is_set():
                logger.info(f"Stop requested during HuggingFaceLocalModel generation loop for {self.model_name}.")
                break

            try:
                # Generate user message
                user_prompt = self._get_user_generation_prompt(medical_text, i)
                
                if self.stop_event and self.stop_event.is_set(): break
                user_message_raw = await asyncio.to_thread(
                    self.pipeline, 
                    user_prompt, 
                    max_length=150,
                    num_return_sequences=1,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                user_message = self._clean_generated_text(user_message_raw[0]['generated_text'])

                if not user_message: continue

                # Generate assistant response
                assistant_prompt = self._get_assistant_generation_prompt(medical_text, user_message)
                
                if self.stop_event and self.stop_event.is_set(): break
                assistant_message_raw = await asyncio.to_thread(
                    self.pipeline, 
                    assistant_prompt, 
                    max_length=config.max_conversation_length,
                    num_return_sequences=1,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                assistant_message = self._clean_generated_text(assistant_message_raw[0]['generated_text'])

                if not assistant_message: continue
                
                # Create ConversationPair
                pair = ConversationPair(
                    user_message=user_message,
                    assistant_message=assistant_message,
                    medical_context=medical_text[:200],
                    confidence_score=0.7,
                    generation_metadata={'model': self.model_name, 'iteration': i}
                )
                all_pairs.append(pair)
                
                await asyncio.sleep(0.05)

            except Exception as e:
                if self.stop_event and self.stop_event.is_set():
                    logger.info(f"HuggingFaceLocalModel generation loop potentially interrupted by stop request: {e}")
                    break
                logger.error(f"Error during local generation with {self.model_name} (iteration {i}): {e}")
        
        return all_pairs
    
    async def _generate_text(self, prompt: str, max_length: int = 100) -> str:
        """Generate text using the local model."""
        try:
            # Run generation in a thread to avoid blocking
            def _generate():
                try:
                    result = self.pipeline(
                        prompt,
                        max_length=len(prompt.split()) + max_length,
                        num_return_sequences=1,
                        temperature=config.temperature,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                    return result[0]['generated_text'] if result else ""
                except Exception as e:
                    logger.error(f"Error in text generation: {e}")
                    return ""
            
            # Run in asyncio thread pool
            import asyncio
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, _generate)
            
        except Exception as e:
            logger.error(f"Error in async text generation: {e}")
            return ""
    
    def _get_user_generation_prompt(self, medical_text: str, variant: int) -> str:
        """Generate prompt for user message creation."""
        prompts = [
            f"Based on this medical information: {medical_text[:500]}, what question might a patient ask?",
            f"Generate a health-related question about: {medical_text[:500]}",
            f"What would someone want to know about: {medical_text[:500]}?",
        ]
        return prompts[variant % len(prompts)]
    
    def _get_assistant_generation_prompt(self, medical_text: str, user_message: str) -> str:
        """Generate prompt for assistant response creation."""
        return f"Medical context: {medical_text[:500]}\n\nPatient question: {user_message}\n\nProvide a helpful, educational response that includes appropriate medical disclaimers:"
    
    def _clean_generated_text(self, text: str) -> str:
        """Clean and format generated text."""
        if not text:
            return ""
        
        # Remove common artifacts
        text = text.strip()
        
        # Remove repeated phrases
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if line and line not in cleaned_lines:
                cleaned_lines.append(line)
        
        result = ' '.join(cleaned_lines)
        
        # Ensure reasonable length
        if len(result) > config.max_conversation_length:
            result = result[:config.max_conversation_length].rsplit(' ', 1)[0]
        
        return result
    
    def _calculate_local_quality_score(self, user_message: str, assistant_message: str) -> float:
        """Calculate quality score for locally generated conversations."""
        score = 0.3  # Lower base score for local models
        
        # Length checks
        if config.min_conversation_length <= len(user_message) <= config.max_conversation_length:
            score += 0.2
        if config.min_conversation_length <= len(assistant_message) <= config.max_conversation_length:
            score += 0.2
        
        # Content quality checks
        medical_terms = ["symptom", "doctor", "pain", "treatment", "medication", "condition", "health", "medical"]
        if any(term in user_message.lower() for term in medical_terms):
            score += 0.1
        if any(term in assistant_message.lower() for term in medical_terms):
            score += 0.1
        
        # Check for appropriate disclaimers in assistant response
        disclaimers = ["consult", "doctor", "medical professional", "healthcare", "advice"]
        if any(disclaimer in assistant_message.lower() for disclaimer in disclaimers):
            score += 0.1
        
        return min(1.0, score)


class AIModelManager:
    """Manager class for coordinating different AI models."""
    
    def __init__(self):
        self.models: Dict[str, BaseAIModel] = {}
        self.stop_event: Optional[asyncio.Event] = None
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize available models based on configuration."""
        
        logger.info(f"Initializing models with API provider: {config.api_provider}")
        logger.info(f"Default model: {config.default_model}")
        logger.info(f"DeepSeek API key configured: {bool(config.deepseek_api_key)}")
        logger.info(f"OpenAI API key configured: {bool(config.openai_api_key)}")
        logger.info(f"Use local model: {config.use_local_model}")
        
        # Initialize based on API provider setting
        if config.api_provider == "deepseek" and config.deepseek_api_key:
            try:
                logger.info("Attempting to initialize DeepSeek model...")
                self.models["deepseek"] = DeepSeekModel(config.default_model)
                logger.info("DeepSeek model initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize DeepSeek model: {e}")
                logger.exception("DeepSeek initialization error details:")
        
        elif config.api_provider == "openai" and config.openai_api_key:
            try:
                logger.info("Attempting to initialize OpenAI-compatible model...")
                logger.info(f"Model name: {config.default_model}")
                logger.info(f"Base URL: {config.openai_base_url}")
                self.models["openai"] = OpenAICompatibleModel(config.default_model)
                logger.info("OpenAI-compatible model initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI-compatible model: {e}")
                logger.exception("OpenAI-compatible initialization error details:")
        
        # Always try to initialize local model as fallback
        if config.use_local_model:
            try:
                logger.info("Attempting to initialize local Hugging Face model...")
                self.models["local"] = HuggingFaceLocalModel()
                logger.info("Local Hugging Face model initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize local model: {e}")
                logger.exception("Local model initialization error details:")
        
        logger.info(f"Total models initialized: {len(self.models)}")
        if self.models:
            logger.info(f"Available models: {list(self.models.keys())}")
            # Test model availability
            for model_key, model in self.models.items():
                try:
                    is_available = model.is_available()
                    logger.info(f"Model {model_key} availability check: {is_available}")
                except Exception as e:
                    logger.error(f"Error checking availability for {model_key}: {e}")
                    logger.exception(f"Availability check error for {model_key}:")
        
        if not self.models:
            logger.error("No AI models could be initialized!")
            raise RuntimeError("No usable AI models available. Please check your API keys and configuration.")
    
    def set_stop_event(self, stop_event: asyncio.Event):
        """Set the stop event for the manager and propagate to its models."""
        self.stop_event = stop_event
        for model in self.models.values():
            if hasattr(model, 'set_stop_event'):
                model.set_stop_event(stop_event)
    
    async def generate_conversations(self, medical_text: str, num_conversations: int) -> List[ConversationPair]:
        """Generate conversations using the best available model."""
        
        if self.stop_event and self.stop_event.is_set():
            logger.info(f"Stop requested before selecting model for provider {config.api_provider}.")
            return []
        
        # Try models in priority order
        model_priority = []
        
        if config.api_provider == "deepseek" and "deepseek" in self.models:
            model_priority.append("deepseek")
        elif config.api_provider == "openai" and "openai" in self.models:
            model_priority.append("openai")
        
        if "local" in self.models:
            model_priority.append("local")
        
        for model_key in model_priority:
            model = self.models[model_key]
            if model.is_available():
                logger.info(f"Generating conversations using {model_key} model")
                try:
                    conversations = await model.generate_conversations(medical_text, num_conversations)
                    if conversations:
                        return conversations
                except Exception as e:
                    logger.error(f"Error with {model_key} model: {e}")
                    continue
        
        logger.error("All models failed to generate conversations")
        return []
    
    def get_available_models(self) -> List[str]:
        """Get list of available model names."""
        available = []
        for key, model in self.models.items():
            if model.is_available():
                available.append(f"{key}:{model.model_name}")
        return available 