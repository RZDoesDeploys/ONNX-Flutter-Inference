# ONNX Image Classification
This repo contains an example inference flutter app that ports a CNN image classifier to Flutter on device.
For this purpose, we convert Googles `mobilenet`, as it is small with only ~3.5M parameters and runs with <200ms on inference on most modern phones and recognizes 1.000 classes.


<img src='assets/example.jpeg' style="float: right; margin-left: 10px;width:200px">

## Getting Started
To convert your model, or any HF model, to the `onnx` format, see the python-notebook `model_to_onnx/convert_model.ipynb`.
Run the flutter application with `flutter run` after installing all dependencies

## References
- Mobilenet [https://huggingface.co/google/mobilenet_v2_1.0_224](https://huggingface.co/google/mobilenet_v2_1.0_224)
- ONNX [https://onnx.ai/](https://onnx.ai/)