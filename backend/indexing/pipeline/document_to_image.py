from pathlib import Path
from typing import List, Dict
import fitz  # PyMuPDF
from logger import logger

class DocumentToImageConverter:
    """Converts PDF pages to images"""

    def __init__(self, output_folder: str = "temp_images"):
        """
        Initialize the converter

        Args:
            output_folder: Folder to save temporary images
        """
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(parents=True, exist_ok=True)

    def pdf_to_images(self, pdf_path: str) -> List[Dict]:
        """
        Convert PDF pages to images

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of dictionaries containing page number and image path
        """
        images = []
        doc = fitz.open(pdf_path)

        for page_num in range(len(doc)):
            page = doc[page_num]
            # Render page to an image (matrix for resolution)
            mat = fitz.Matrix(2, 2)  # 2x zoom for better quality
            pix = page.get_pixmap(matrix=mat)

            # Save image
            img_path = self.output_folder / f"{Path(pdf_path).stem}_page_{page_num + 1}.png"
            pix.save(str(img_path))

            images.append({
                'page_number': page_num + 1,
                'image_path': str(img_path),
                'source': Path(pdf_path).name
            })

        doc.close()
        return images

    def convert_document(self, file_path: str) -> List[Dict]:
        """
        Convert PDF document to images

        Args:
            file_path: Path to the PDF document

        Returns:
            List of dictionaries with page info and image paths
        """
        file_path_obj = Path(file_path)
        extension = file_path_obj.suffix.lower()

        if extension == '.pdf':
            return self.pdf_to_images(file_path)
        else:
            logger.warning(f"Unsupported file type: {extension}. Only PDF files are supported.")
            logger.warning("Please convert your document to PDF first.")
            return []

    def cleanup(self):
        """Remove all temporary images"""
        for img_file in self.output_folder.glob('*.png'):
            img_file.unlink()
        logger.info(f"Cleaned up temporary images from {self.output_folder}")
