import re
from typing import Optional, List, Dict, Any
from datetime import datetime

# Regex patterns for intelligent document field extraction
PATTERNS = {
    "invoice_number": [
        r'(?i)(?:invoice\s*(?:number|num|no|#)|inv\s*(?:number|num|no|#)|bill\s*(?:number|num|no|#)|b\.?\s*no\.?|memo\s*(?:no|#)?|ref\s*(?:number|num|no|#))[\s.:\-_]*([A-Z0-9\-_/]{2,25})',
        r'(?i)(?:invoice|inv|bill|receipt|factura)[\s#.:\-_]+([A-Z0-9\-_/]{2,25})',
        r'(?i)#\s*([A-Z0-9\-_]{3,20})'
    ],
    "date": [
        # YYYY-MM-DD, YYYY/MM/DD, YYYY.MM.DD
        r'\b(20\d{2}[-/.]\d{1,2}[-/.]\d{1,2})\b',
        # DD-MM-YYYY, DD/MM/YYYY, DD.MM.YYYY or D-M-YY / D-M-YYYY
        r'\b(\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4})\b',
        # Month DD, YYYY (e.g., January 15, 2024 or Jan 15, 2024)
        r'\b((?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2},?\s+\d{2,4})\b',
        # DD Month YYYY (e.g., 15 Jan 2024)
        r'\b(\d{1,2}\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember))\s+\d{2,4})\b'
    ],
    "total_amount": [
        r'(?i)(?:total|grand\s*total|balance\s*due|amount\s*due|net\s*payable|total\s*usd|total\s*eur)[\s:$€£₹]*([\d,]+(?:\.\d{1,2})?)\b',
        r'(?i)(?:total|amount\s*due)[\s:]*([$€£₹]\s*[\d,]+(?:\.\d{1,2})?)\b'
    ],
    "subtotal": [
        r'(?i)(?:subtotal|sub\s*total|net\s*amount)[\s:$€£₹]*([\d,]+(?:\.\d{1,2})?)\b'
    ],
    "tax_amount": [
        r'(?i)(?:tax|vat|gst|sales\s*tax|hst)[\s:(%0-9)]*[\s:$€£₹]*([\d,]+(?:\.\d{1,2})?)\b'
    ],
    "customer_name": [
        r'(?i)(?:customer\s*name|client\s*name|bill\s*to|sold\s*to|m/s|name)[\s.:\-_]+([A-Za-z0-9\s.\-_/]{3,50})'
    ],
    "bank_name": [
        r'(?i)(?:bank\s*name|bank)[\s.:\-_]+([A-Za-z0-9\s.()]{3,40})'
    ],
    "account_number": [
        r'(?i)(?:a/c\s*(?:no|num|number|numer)?|account\s*(?:no|number)?)[\s.:\-_]*([0-9]{8,20})'
    ],
    "ifsc_code": [
        r'(?i)(?:ifsc|ifsc\s*code)[\s.:\-_]*([A-Z]{4}0[A-Z0-9]{6})'
    ],
    "email": [
        r'\b([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)\b'
    ],
    "phone": [
        r'(\+?\d{1,3}[-.\s]?(?:\(?\d{3}\)?[-.\s]?|\d{3}[-.\s]?)\d{3}[-.\s]?\d{4})',
        r'(?i)(?:mo|mobile|phone|tel)[\s.:\-_]*(\+?[0-9]{10,14})',
        r'(\+?[0-9]{10,14})'
    ],
    "currency": [
        r'([$€£₹]|USD|EUR|GBP|INR|CAD|AUD)'
    ]
}

def clean_ocr_text(text: str) -> str:
    """Cleans unnecessary whitespaces and common OCR artifacts."""
    if not text:
        return ""
    # Normalize multiple spaces/newlines
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Strip leading/trailing control characters
    text = text.strip()
    return text

def parse_amount(text_val: str) -> Optional[float]:
    """Extracts numeric float from a string like '$ 1,250.50'."""
    if not text_val:
        return None
    # Remove currency signs and extra spaces
    cleaned = re.sub(r'[^0-9.]', '', text_val.replace(',', ''))
    try:
        return float(cleaned)
    except (ValueError, TypeError):
        return None

def standardize_date(date_str: str) -> Optional[str]:
    """Attempts to standardize various date formats into YYYY-MM-DD."""
    if not date_str:
        return None
    date_str = date_str.strip()

    formats = [
        "%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d",
        "%d-%m-%Y", "%d/%m/%Y", "%d.%m.%Y",
        "%m-%d-%Y", "%m/%d/%Y",
        "%B %d, %Y", "%b %d, %Y", "%B %d %Y", "%b %d %Y",
        "%d %B %Y", "%d %b %Y"
    ]

    for fmt in formats:
        try:
            dt = datetime.strptime(date_str, fmt)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            continue

    return date_str

def correct_numeric_ocr_errors(text: str) -> str:
    """
    Substitutes common OCR letter misrecognitions in numeric tokens:
    e.g., 'O' or 'o' -> '0', 'l' or 'I' -> '1', 'S' or 's' -> '5', 'B' -> '8'.
    """
    # Only applies when token looks like an intended number
    if re.search(r'\d', text):
        trans = str.maketrans({
            'O': '0', 'o': '0',
            'I': '1', 'l': '1',
            'S': '5', 's': '5',
            'B': '8'
        })
        return text.translate(trans)
    return text
