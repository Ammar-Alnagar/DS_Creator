import unittest
from pathlib import Path
from src.pdf_processor import PDFProcessor

class TestPDFProcessor(unittest.TestCase):
    def setUp(self):
        self.processor = PDFProcessor()
        self.test_pdf_path = Path("test_data/test.pdf")
        self.test_pdf_path.parent.mkdir(exist_ok=True)
        # Create a dummy PDF for testing
        from fpdf import FPDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="This is a test PDF document.", ln=1, align="C")
        pdf.output(self.test_pdf_path)

    def test_extract_text_from_pdf(self):
        # Create a dummy PDF for testing
        from fpdf import FPDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="This is a test PDF document.", ln=1, align="C")
        pdf.output(self.test_pdf_path)

        chunks = self.processor.extract_text_from_pdf(self.test_pdf_path)
        self.assertGreater(len(chunks), 0)
        self.assertIn("This is a test PDF document.", chunks[0].content)

if __name__ == "__main__":
    unittest.main()
