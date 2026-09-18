# Custom Text Recognition Models Directory

Place your custom text recognition model weights (e.g. CRNN, TrOCR, SVTR, or custom PyTorch / ONNX models) and vocabularies here.

### Standard Configuration:
- `recognizer.onnx` or `weights.pth`: Model weights for sequence recognition.
- `vocab.txt` or `charset.json`: Character dictionary mapping indices to unicode characters.
- `config.json`: Model architecture, input image dimensions (e.g., 32x100), normalization mean/std.

The `app.ocr.recognizer.CustomRecognizer` class checks this directory first. If custom weights are not found, it seamlessly delegates to the system recognizer or built-in CV recognition engine.
