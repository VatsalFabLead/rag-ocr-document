from pathlib import Path
from typing import List, Tuple
import pypdfium2 as pdfium
from PIL import Image

def get_pdf_page_count(pdf_path: str) -> int:
    """Returns the total number of pages in a PDF document."""
    pdf = pdfium.PdfDocument(pdf_path)
    count = len(pdf)
    pdf.close()
    return count

def convert_pdf_to_images(pdf_path: str, output_dir: str, dpi: int = 300) -> List[Tuple[int, str, int, int]]:
    """
    Renders each page of a PDF document into a high-resolution PNG image.
    Returns a list of tuples: (page_number, image_path, width, height).
    """
    pdf = pdfium.PdfDocument(pdf_path)
    page_results = []
    scale = dpi / 72.0  # 72 points per inch in PDF coordinate space

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for i, page in enumerate(pdf):
        page_number = i + 1
        # Render page to bitmap at specified scale
        bitmap = page.render(scale=scale)
        pil_image = bitmap.to_pil()
        width, height = pil_image.size

        img_filename = f"page_{page_number}.png"
        img_path = str(output_path / img_filename)
        pil_image.save(img_path, format="PNG")
        page_results.append((page_number, img_path, width, height))

    pdf.close()
    return page_results
