import unittest
from unittest.mock import patch, AsyncMock
from src.ai_models import AIModelManager, DeepSeekModel, OpenAICompatibleModel, HuggingFaceLocalModel

class TestAIModels(unittest.TestCase):
    @patch("src.ai_models.DeepSeekModel.is_available", return_value=True)
    @patch("src.ai_models.OpenAICompatibleModel.is_available", return_value=True)
    @patch("src.ai_models.HuggingFaceLocalModel.is_available", return_value=True)
    def test_ai_model_manager_initialization(self, mock_hf, mock_openai, mock_deepseek):
        with patch("config.config.api_provider", "deepseek"), \
             patch("config.config.deepseek_api_key", "test_key"):
            manager = AIModelManager()
            self.assertIn("deepseek", manager.models)

        with patch("config.config.api_provider", "openai"), \
             patch("config.config.openai_api_key", "test_key"):
            manager = AIModelManager()
            self.assertIn("openai", manager.models)

        with patch("config.config.use_local_model", True):
            with patch("src.ai_models.HuggingFaceLocalModel._load_model"):
                manager = AIModelManager()
                self.assertIn("local", manager.models)

if __name__ == "__main__":
    unittest.main()
