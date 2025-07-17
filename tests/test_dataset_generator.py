import unittest
from unittest.mock import patch, MagicMock, AsyncMock
from src.dataset_generator import MedicalDatasetGenerator, GenerationStats
from src.ai_models import ConversationPair
from src.pdf_processor import TextChunk

class TestDatasetGenerator(unittest.TestCase):
    @patch("src.dataset_generator.PDFProcessor")
    @patch("src.dataset_generator.AIModelManager")
    @patch("src.dataset_generator.HuggingFaceUploader")
    def test_generate_dataset(self, mock_uploader, mock_ai_manager, mock_pdf_processor):
        # Arrange
        mock_pdf_processor.return_value.process_directory.return_value = {
            "test.pdf": [TextChunk("content", 1, 1, "test.pdf", "test_method")]
        }
        mock_ai_manager.return_value.generate_conversations = AsyncMock(return_value=[
            ConversationPair("user_msg", "asst_msg", "context", 0.9, {})
        ])

        generator = MedicalDatasetGenerator()

        # Act
        import asyncio
        from pathlib import Path
        # Create a dummy input directory
        input_dir = Path("test_data/input")
        input_dir.mkdir(exist_ok=True, parents=True)
        (input_dir / "dummy.pdf").touch()

        output_path = asyncio.run(generator.generate_dataset(input_dir=input_dir))

        # Assert
        self.assertTrue(output_path.exists())
        with open(output_path, "r") as f:
            data = [json.loads(line) for line in f]
            self.assertEqual(len(data), 1)
            self.assertEqual(data[0]["user_message"], "user_msg")

if __name__ == "__main__":
    unittest.main()
