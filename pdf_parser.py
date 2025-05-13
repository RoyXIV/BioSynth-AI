## Step 3: PDF Parsing (pdf_parser.py)

import fitz  # PyMuPDF

def extract_text_sections(pdf_path):
    doc = fitz.open(pdf_path)
    text = "\n".join([page.get_text() for page in doc])
    # You can split text into sections like 'Methods', 'Results' later
    return text
