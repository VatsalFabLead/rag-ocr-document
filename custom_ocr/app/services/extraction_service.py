import re
from typing import Dict, List, Optional, Any
from app.models.extracted_field import (
    ExtractedField, FieldType, KeyValuePair, DocumentExtractionResult
)
from app.models.ocr_result import PageOCRResult, LineBox, WordBox, BoundingBox
from app.utils.text_utils import (
    PATTERNS, parse_amount, standardize_date, clean_ocr_text
)

class ExtractionService:
    """
    Step 5 — Automatic Field Extraction Service
    - Rule-based & Regex Extraction
    - Spatial Key-Value Pair Extraction
    - Pluggable ML / NLP / LLM Structured Extraction
    """

    def extract(
        self,
        doc_id: str,
        pages: List[PageOCRResult],
        doc_type_hint: Optional[str] = None
    ) -> DocumentExtractionResult:
        full_text = "\n".join([p.full_text for p in pages])
        fields: Dict[str, ExtractedField] = {}
        key_values: List[KeyValuePair] = []

        # 1. Regex & Rule-Based Field Extraction
        self._extract_rule_fields(pages, fields, full_text)

        # 2. Spatial Key-Value Pair Extraction
        for p in pages:
            kv_pairs = self._extract_spatial_key_values(p)
            key_values.extend(kv_pairs)

        # 3. Consolidate key-values into primary fields if not already found
        self._merge_key_values_into_fields(key_values, fields)

        # Collect all tables across pages
        all_tables = [table for p in pages for table in p.tables]

        # Calculate average confidence
        field_confs = [f.confidence for f in fields.values()]
        avg_conf = sum(field_confs) / len(field_confs) if field_confs else 0.85

        doc_type = doc_type_hint or self._infer_document_type(full_text)

        return DocumentExtractionResult(
            doc_id=doc_id,
            document_type=doc_type,
            fields=fields,
            key_values=key_values,
            tables=all_tables,
            summary=self._generate_summary(doc_type, fields),
            raw_text=full_text,
            overall_confidence=round(avg_conf, 3)
        )

    def _infer_document_type(self, text: str) -> str:
        text_lower = text.lower()
        if any(k in text_lower for k in ["invoice", "tax invoice", "bill to", "due date", "bill of supply", "debit memo", "cash memo"]):
            return "invoice"
        elif any(k in text_lower for k in ["receipt", "pos", "terminal", "subtotal"]):
            return "receipt"
        elif any(k in text_lower for k in ["identity", "license", "passport", "dob", "pan", "aadhaar"]):
            return "id_card"
        return "generic"

    def _extract_rule_fields(
        self, pages: List[PageOCRResult], fields: Dict[str, ExtractedField], full_text: str = ""
    ):
        # Vendor Name (look in top lines for business names)
        if "vendor_name" not in fields:
            for page in pages:
                for line in page.lines[:10]:
                    txt = line.text.strip()
                    if any(w in txt.lower() for w in ["traders", "store", "shop", "enterprise", "ltd", "pvt", "company", "market", "corp", "agency", "inc"]):
                        clean_v = re.sub(r'^[^A-Za-z0-9]+', '', txt)
                        if len(clean_v) >= 4:
                            fields["vendor_name"] = ExtractedField(
                                field_name="Vendor Name",
                                value=clean_v,
                                raw_text=txt,
                                field_type=FieldType.VENDOR_NAME,
                                confidence=0.91,
                                page_number=page.page_number,
                                bbox=line.bbox,
                                extraction_method="rule_based"
                            )
                            break

        for page in pages:
            # Check each line for targeted patterns
            for line in page.lines:
                text = line.text.strip()

                # Invoice / Bill Number
                if "invoice_number" not in fields:
                    for pat in PATTERNS["invoice_number"]:
                        match = re.search(pat, text)
                        if match:
                            val = match.group(1).strip()
                            if val.lower() in ["number", "no", "num", "id", "date", "total", "code"]:
                                continue
                            fields["invoice_number"] = ExtractedField(
                                field_name="Invoice Number",
                                value=val,
                                raw_text=match.group(0),
                                field_type=FieldType.INVOICE_NUMBER,
                                confidence=0.92,
                                page_number=page.page_number,
                                bbox=line.bbox,
                                extraction_method="regex"
                            )
                            break

                # Date
                if "invoice_date" not in fields:
                    for pat in PATTERNS["date"]:
                        match = re.search(pat, text)
                        if match:
                            raw_date = match.group(1)
                            std_date = standardize_date(raw_date)
                            fields["invoice_date"] = ExtractedField(
                                field_name="Invoice Date",
                                value=std_date,
                                raw_text=raw_date,
                                field_type=FieldType.INVOICE_DATE,
                                confidence=0.90,
                                page_number=page.page_number,
                                bbox=line.bbox,
                                extraction_method="regex"
                            )
                            break

                # Total Amount
                if "total_amount" not in fields:
                    for pat in PATTERNS["total_amount"]:
                        match = re.search(pat, text)
                        if match:
                            raw_amt = match.group(1)
                            parsed = parse_amount(raw_amt)
                            if parsed is not None:
                                fields["total_amount"] = ExtractedField(
                                    field_name="Total Amount",
                                    value=parsed,
                                    raw_text=raw_amt,
                                    field_type=FieldType.TOTAL_AMOUNT,
                                    confidence=0.94,
                                    page_number=page.page_number,
                                    bbox=line.bbox,
                                    extraction_method="regex"
                                )
                                break

                # Balance Amount
                if "balance_amount" not in fields:
                    bal_match = re.search(r'(?i)(?:balance|bal)[\s\S]{0,15}?([\d,]+(?:\.\d{1,2})?)\b', text)
                    if bal_match:
                        b_amt = parse_amount(bal_match.group(1))
                        if b_amt and b_amt > 0:
                            fields["balance_amount"] = ExtractedField(
                                field_name="Balance Amount",
                                value=b_amt,
                                raw_text=bal_match.group(0),
                                field_type=FieldType.BALANCE_AMOUNT,
                                confidence=0.89,
                                page_number=page.page_number,
                                bbox=line.bbox,
                                extraction_method="regex"
                            )

                # Subtotal
                if "subtotal" not in fields:
                    for pat in PATTERNS["subtotal"]:
                        match = re.search(pat, text)
                        if match:
                            raw_amt = match.group(1)
                            parsed = parse_amount(raw_amt)
                            if parsed is not None:
                                fields["subtotal"] = ExtractedField(
                                    field_name="Subtotal",
                                    value=parsed,
                                    raw_text=raw_amt,
                                    field_type=FieldType.SUBTOTAL,
                                    confidence=0.88,
                                    page_number=page.page_number,
                                    bbox=line.bbox,
                                    extraction_method="regex"
                                )
                                break

                # Tax / VAT
                if "tax_amount" not in fields:
                    for pat in PATTERNS["tax_amount"]:
                        match = re.search(pat, text)
                        if match:
                            raw_amt = match.group(1)
                            parsed = parse_amount(raw_amt)
                            if parsed is not None:
                                fields["tax_amount"] = ExtractedField(
                                    field_name="Tax Amount",
                                    value=parsed,
                                    raw_text=raw_amt,
                                    field_type=FieldType.TAX_AMOUNT,
                                    confidence=0.88,
                                    page_number=page.page_number,
                                    bbox=line.bbox,
                                    extraction_method="regex"
                                )
                                break

                # Customer Name
                if "customer_name" not in fields and "customer_name" in PATTERNS:
                    for pat in PATTERNS["customer_name"]:
                        match = re.search(pat, text)
                        if match:
                            val = match.group(1).strip()
                            if len(val) >= 3 and val.lower() not in ["number", "date", "address", "phone"]:
                                fields["customer_name"] = ExtractedField(
                                    field_name="Customer Name",
                                    value=val,
                                    raw_text=match.group(0),
                                    field_type=FieldType.CUSTOMER_NAME,
                                    confidence=0.88,
                                    page_number=page.page_number,
                                    bbox=line.bbox,
                                    extraction_method="regex"
                                )
                                break

                # Bank Name
                if "bank_name" not in fields and "bank_name" in PATTERNS:
                    for pat in PATTERNS["bank_name"]:
                        match = re.search(pat, text)
                        if match:
                            val = match.group(1).strip()
                            if len(val) >= 3:
                                fields["bank_name"] = ExtractedField(
                                    field_name="Bank Name",
                                    value=val,
                                    raw_text=match.group(0),
                                    field_type=FieldType.GENERIC_KEY_VALUE,
                                    confidence=0.89,
                                    page_number=page.page_number,
                                    bbox=line.bbox,
                                    extraction_method="regex"
                                )
                                break

                # Account Number
                if "account_number" not in fields and "account_number" in PATTERNS:
                    for pat in PATTERNS["account_number"]:
                        match = re.search(pat, text)
                        if match:
                            fields["account_number"] = ExtractedField(
                                field_name="Account Number",
                                value=match.group(1).strip(),
                                raw_text=match.group(0),
                                field_type=FieldType.GENERIC_KEY_VALUE,
                                confidence=0.91,
                                page_number=page.page_number,
                                bbox=line.bbox,
                                extraction_method="regex"
                            )
                            break

                # IFSC Code
                if "ifsc_code" not in fields and "ifsc_code" in PATTERNS:
                    for pat in PATTERNS["ifsc_code"]:
                        match = re.search(pat, text)
                        if match:
                            fields["ifsc_code"] = ExtractedField(
                                field_name="IFSC Code",
                                value=match.group(1).strip(),
                                raw_text=match.group(0),
                                field_type=FieldType.GENERIC_KEY_VALUE,
                                confidence=0.95,
                                page_number=page.page_number,
                                bbox=line.bbox,
                                extraction_method="regex"
                            )
                            break

                # Email
                if "email" not in fields:
                    match = re.search(PATTERNS["email"][0], text)
                    if match:
                        fields["email"] = ExtractedField(
                            field_name="Email Address",
                            value=match.group(1),
                            raw_text=match.group(0),
                            field_type=FieldType.EMAIL,
                            confidence=0.96,
                            page_number=page.page_number,
                            bbox=line.bbox,
                            extraction_method="regex"
                        )

                # Phone
                if "phone" not in fields:
                    for pat in PATTERNS["phone"]:
                        match = re.search(pat, text)
                        if match:
                            fields["phone"] = ExtractedField(
                                field_name="Phone Number",
                                value=match.group(1).strip(),
                                raw_text=match.group(0),
                                field_type=FieldType.PHONE,
                                confidence=0.85,
                                page_number=page.page_number,
                                bbox=line.bbox,
                                extraction_method="regex"
                            )
                            break

        # Total Amount full-text cross-line fallback
        if "total_amount" not in fields and full_text:
            tot_match = re.search(r'(?i)(?:total|grand\s*total|net\s*payable)[\s\S]{0,60}?([\d,]+\.\d{2})\b', full_text)
            if tot_match:
                t_amt = parse_amount(tot_match.group(1))
                if t_amt and t_amt > 0:
                    fields["total_amount"] = ExtractedField(
                        field_name="Total Amount",
                        value=t_amt,
                        raw_text=tot_match.group(0),
                        field_type=FieldType.TOTAL_AMOUNT,
                        confidence=0.92,
                        page_number=1,
                        bbox=None,
                        extraction_method="regex"
                    )

    def _extract_spatial_key_values(self, page: PageOCRResult) -> List[KeyValuePair]:
        """
        Extracts key-value pairs from:
        1. Line-level delimiters (e.g. 'Name: John', 'Bill No: 4307')
        2. Spatial word proximity (label ending with ':' and candidate values to the right or below)
        """
        pairs: List[KeyValuePair] = []
        seen_keys = set()

        # 1. Line-level Key-Value / Form Field parsing
        for line in page.lines:
            text = line.text.strip()
            # Support :, =, and - delimiters
            delim_match = re.search(r'[:=]| - ', text)
            if delim_match:
                d_start = delim_match.start()
                d_end = delim_match.end()
                k_candidate = text[:d_start].strip()
                v_candidate = text[d_end:].strip()
                # Clean up punctuation and reject empty or noisy tokens
                k_clean = re.sub(r'^[^A-Za-z0-9]+|[^A-Za-z0-9\s.]+$', '', k_candidate).strip()
                v_clean = re.sub(r'^[^A-Za-z0-9]+|[^A-Za-z0-9\s.,]+$', '', v_candidate).strip()
                if 2 <= len(k_clean) <= 35 and len(v_clean) > 0 and k_clean.lower() not in seen_keys:
                    seen_keys.add(k_clean.lower())
                    pairs.append(KeyValuePair(
                        key=k_clean,
                        value=v_clean,
                        confidence=0.90,
                        page_number=page.page_number,
                        key_bbox=line.bbox,
                        value_bbox=line.bbox
                    ))

        # 2. Spatial proximity word parsing
        words = page.words
        for i, word in enumerate(words):
            text = word.text.strip()
            if text.endswith(":") and len(text) > 1:
                key_text = text[:-1].strip()
                key_bbox = word.bbox
                if not key_bbox or key_text.lower() in seen_keys:
                    continue

                # Adaptive line height threshold (proportional to word height)
                dy = max(20, int(key_bbox.height * 1.6))

                # Look for candidate words to the right on the same row
                candidates_right = [
                    w for w in words[i+1:]
                    if abs(w.bbox.y - key_bbox.y) < dy and w.bbox.x > key_bbox.x
                ]

                if candidates_right:
                    # Collect contiguous value words to the right
                    val_words = [candidates_right[0]]
                    for next_w in candidates_right[1:]:
                        prev_w = val_words[-1]
                        gap = next_w.bbox.x - (prev_w.bbox.x + prev_w.bbox.width)
                        if gap < max(50, prev_w.bbox.width * 1.5):
                            val_words.append(next_w)
                        else:
                            break
                    val_text = " ".join([w.text for w in val_words]).strip()
                    if val_text:
                        seen_keys.add(key_text.lower())
                        pairs.append(KeyValuePair(
                            key=key_text,
                            value=val_text,
                            confidence=0.87,
                            page_number=page.page_number,
                            key_bbox=key_bbox,
                            value_bbox=val_words[-1].bbox
                        ))
                else:
                    # Look for candidate word directly below (form layout)
                    candidates_below = [
                        w for w in words[i+1:]
                        if abs(w.bbox.x - key_bbox.x) < max(50, key_bbox.width) and 0 < (w.bbox.y - key_bbox.y) < (dy * 3)
                    ]
                    if candidates_below:
                        val_word = candidates_below[0]
                        seen_keys.add(key_text.lower())
                        pairs.append(KeyValuePair(
                            key=key_text,
                            value=val_word.text.strip(),
                            confidence=0.82,
                            page_number=page.page_number,
                            key_bbox=key_bbox,
                            value_bbox=val_word.bbox
                        ))

        return pairs

    def _merge_key_values_into_fields(
        self, key_values: List[KeyValuePair], fields: Dict[str, ExtractedField]
    ):
        for kv in key_values:
            k_lower = kv.key.lower()
            val = kv.value.strip()

            if any(k in k_lower for k in ["invoice", "inv", "bill", "b.no", "bno"]) and "invoice_number" not in fields:
                fields["invoice_number"] = ExtractedField(
                    field_name="Invoice Number",
                    value=val,
                    raw_text=val,
                    field_type=FieldType.INVOICE_NUMBER,
                    confidence=kv.confidence,
                    page_number=kv.page_number,
                    bbox=kv.value_bbox,
                    extraction_method="spatial_kv"
                )
            elif any(k in k_lower for k in ["date", "dt"]) and "invoice_date" not in fields:
                fields["invoice_date"] = ExtractedField(
                    field_name="Invoice Date",
                    value=standardize_date(val),
                    raw_text=val,
                    field_type=FieldType.INVOICE_DATE,
                    confidence=kv.confidence,
                    page_number=kv.page_number,
                    bbox=kv.value_bbox,
                    extraction_method="spatial_kv"
                )
            elif any(k in k_lower for k in ["total", "10tal", "net amount", "grand total", "payable", "balance"]) and "total_amount" not in fields:
                amt = parse_amount(val)
                if amt is not None and amt > 0:
                    fields["total_amount"] = ExtractedField(
                        field_name="Total Amount",
                        value=amt,
                        raw_text=val,
                        field_type=FieldType.TOTAL_AMOUNT,
                        confidence=kv.confidence,
                        page_number=kv.page_number,
                        bbox=kv.value_bbox,
                        extraction_method="spatial_kv"
                    )
            elif any(k in k_lower for k in ["name", "customer", "buyer", "sold to"]) and "customer_name" not in fields:
                fields["customer_name"] = ExtractedField(
                    field_name="Customer Name",
                    value=val,
                    raw_text=val,
                    field_type=FieldType.CUSTOMER_NAME,
                    confidence=kv.confidence,
                    page_number=kv.page_number,
                    bbox=kv.value_bbox,
                    extraction_method="spatial_kv"
                )
            elif "bank" in k_lower and "bank_name" not in fields:
                fields["bank_name"] = ExtractedField(
                    field_name="Bank Name",
                    value=val,
                    raw_text=val,
                    field_type=FieldType.GENERIC_KEY_VALUE,
                    confidence=kv.confidence,
                    page_number=kv.page_number,
                    bbox=kv.value_bbox,
                    extraction_method="spatial_kv"
                )
            elif ("a/c" in k_lower or "account" in k_lower) and "account_number" not in fields:
                fields["account_number"] = ExtractedField(
                    field_name="Account Number",
                    value=val,
                    raw_text=val,
                    field_type=FieldType.GENERIC_KEY_VALUE,
                    confidence=kv.confidence,
                    page_number=kv.page_number,
                    bbox=kv.value_bbox,
                    extraction_method="spatial_kv"
                )
            elif "ifsc" in k_lower and "ifsc_code" not in fields:
                fields["ifsc_code"] = ExtractedField(
                    field_name="IFSC Code",
                    value=val,
                    raw_text=val,
                    field_type=FieldType.GENERIC_KEY_VALUE,
                    confidence=kv.confidence,
                    page_number=kv.page_number,
                    bbox=kv.value_bbox,
                    extraction_method="spatial_kv"
                )
            elif any(k in k_lower for k in ["mo", "mobile", "phone"]) and "phone" not in fields:
                fields["phone"] = ExtractedField(
                    field_name="Phone Number",
                    value=val,
                    raw_text=val,
                    field_type=FieldType.PHONE,
                    confidence=kv.confidence,
                    page_number=kv.page_number,
                    bbox=kv.value_bbox,
                    extraction_method="spatial_kv"
                )
            else:
                # Generic custom form field promotion
                field_key = re.sub(r'[^a-zA-Z0-9_]', '_', kv.key.lower().strip()).strip('_')
                if field_key and field_key not in fields and len(field_key) <= 30:
                    fields[field_key] = ExtractedField(
                        field_name=kv.key.strip(),
                        value=val,
                        raw_text=val,
                        field_type=FieldType.GENERIC_KEY_VALUE,
                        confidence=kv.confidence,
                        page_number=kv.page_number,
                        bbox=kv.value_bbox,
                        extraction_method="form_field"
                    )

    def _generate_summary(self, doc_type: str, fields: Dict[str, ExtractedField]) -> str:
        parts = [f"Classified as {doc_type.upper()}."]
        if "invoice_number" in fields:
            parts.append(f"Number: {fields['invoice_number'].value}.")
        if "customer_name" in fields:
            parts.append(f"Name: {fields['customer_name'].value}.")
        if "invoice_date" in fields:
            parts.append(f"Date: {fields['invoice_date'].value}.")
        if "total_amount" in fields:
            parts.append(f"Total: {fields['total_amount'].value}.")
        return " ".join(parts)
