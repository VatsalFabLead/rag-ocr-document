# Custom Text Detection Models Directory

Place your custom text detection model weights (e.g. DBNet, CRAFT, EAST, or custom PyTorch / ONNX models) here.

### Standard Configuration:
- `model.onnx` or `weights.pth`: Serialized neural network model weights.
- `config.json`: Detection hyperparameters (e.g. input resolution, text threshold, box threshold).

The `app.ocr.detector.CustomDetector` class can load custom weights from this directory. When no custom weights are present, the system uses the built-in morphological contour detector or configured system engine.
