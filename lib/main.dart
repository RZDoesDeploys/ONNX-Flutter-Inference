import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:onnx_image_classification/decode_clases.dart';
import 'package:onnx_image_classification/output_result.dart';
import 'package:onnxruntime/onnxruntime.dart';
import 'package:flutter/services.dart';
import 'package:image_picker/image_picker.dart';
import 'dart:io';
import 'package:image/image.dart' as img;
import 'dart:collection';

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'ONNX Image Classification',
      theme: ThemeData(
        primarySwatch: Colors.blue,
        primaryColor: Colors.brown
      ),
      home: ImageClassificationPage(),
    );
  }
}

class ImageClassificationPage extends StatefulWidget {
  @override
  _ImageClassificationPageState createState() =>
      _ImageClassificationPageState();
}

class _ImageClassificationPageState extends State<ImageClassificationPage> {
  OrtSession? _session;
  File? _imageFile;
  final ImagePicker _picker = ImagePicker();
  final ValueNotifier<List<OutputResult>> predictions = ValueNotifier([]);
  final ValueNotifier<int> inferenceTime = ValueNotifier(0);

  @override
  void initState() {
    super.initState();
    _loadModel();
  }

  // Load the ONNX model from the assets folder
  Future<void> _loadModel() async {
    try {
      OrtEnv.instance.init();
      final sessionOptions = OrtSessionOptions();
      const assetFileName = 'assets/mobile.onnx';
      final rawAssetFile = await rootBundle.load(assetFileName);
      final bytes = rawAssetFile.buffer.asUint8List();

      _session = OrtSession.fromBuffer(bytes, sessionOptions);
      print('ONNX Model loaded successfully');
    } catch (e) {
      print("Error loading model: $e");
    }
  }

  // Pick image from gallery or camera
  Future<void> _pickImage(ImageSource source) async {
    final pickedFile = await _picker.pickImage(source: source);
    if (pickedFile != null) {
      setState(() {
        _imageFile = File(pickedFile.path);
      });
      _runInference(_imageFile!);
    }
  }

Float32List _preprocessImage(File imageFile) {
    // Read image and resize
    final Uint8List imageBytes = imageFile.readAsBytesSync();
    img.Image? image = img.decodeImage(imageBytes);
    img.Image resizedImage = img.copyResize(image!, width: 224, height: 224);

    // Fomat image bytes to list to later reshape correctly to tensor
    const int channels = 3;
    final floatList = Float32List(resizedImage.width * resizedImage.height * channels);
    for (int x = 0; x < resizedImage.width; x++) {
      for (int y = 0; y < resizedImage.height; y++) {
          final pixel = resizedImage.getPixel(x, y);
          for (int i = 0; i < channels; i++){
            floatList[(x + (y* resizedImage.width)) + (resizedImage.width * resizedImage.height * i)] = pixel[i] / 255;
          } 
        }
      }
    return floatList;
  }

  List<OutputResult> processOutputs(List<OrtValue?>? outputs, int topK) {
    if (outputs == null) return [];

    for (var ortValue in outputs) {
      var output = ortValue?.value as List<List<double>>?;
      if (output == null || output.isEmpty) continue;

      for (var innerList in output) {
        if (innerList.isEmpty) continue;

        // Sort indices by their values in descending order and take the topK
        List<OutputResult> topResults =  List.generate(
                  innerList.length,
                  (index) => OutputResult(
                    id: index,
                    className: ClassNameProvider.getClassName(index),
                    confidence: innerList[index],
                  )
                )
                ..sort((a, b) => b.confidence.compareTo(a.confidence))
                ..toList();
        return topResults.take(topK).toList();
      }
    }
    return [];
  }

  // Run inference on the selected image
  Future<void> _runInference(File imageFile) async {
    if (_session == null) {
      print("ONNX model not loaded yet");
      return;
    }

    // Preprocess the image
    Float32List imageData = _preprocessImage(imageFile);
    final inputShape = [1, 3, 224, 224];  // [batch_size, channels, height, width]
    final inputTensor = OrtValueTensor.createTensorWithDataList(
        imageData, inputShape);
    print("inputTensor");
    print(inputTensor);
    final inputs = {'input': inputTensor};

    // Perform inference
    try {
      final stopwatch = Stopwatch()..start();
      final runOptions = OrtRunOptions();
      final outputs = await _session!.runAsync(runOptions, inputs);
      predictions.value = processOutputs(outputs, 5);
      inferenceTime.value = stopwatch.elapsedMilliseconds;
      // Release resources
      inputTensor.release();
      runOptions.release();
    } catch (e) {
      print("Error during inference: $e");
    }
  }

  @override
  void dispose() {
    OrtEnv.instance.release();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('ONNX Image Classification'),
        centerTitle: true,
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.start,
          children: <Widget>[
              SizedBox(
                height: MediaQuery.of(context).size.height * 0.4,
                width: MediaQuery.of(context).size.width,
                child: _imageFile == null 
                          ? IconButton(icon: Icon(
                                          Icons.image,
                                          size: 100,
                                          color: Theme.of(context).primaryColor,
                                        ),
                                      color: Theme.of(context).primaryColor,
                                      onPressed: () => _pickImage(ImageSource.gallery),)
                          : Image.file(
                              _imageFile!,
                              fit: BoxFit.cover,
                            ),
              ),              
              const SizedBox(height: 10),   
              Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  ElevatedButton.icon(
                    onPressed: () => _pickImage(ImageSource.gallery),
                    icon: const Icon(Icons.photo_library),
                    label: const Text('Gallery'),
                    style: ElevatedButton.styleFrom(
                      padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 12),
                      foregroundColor: Theme.of(context).primaryColor
                    ),
                  ),
                  const SizedBox(width: 10),
                  ElevatedButton.icon(
                    onPressed: () => _pickImage(ImageSource.camera),
                    icon: const Icon(Icons.camera_alt),
                    label: const Text('Camera'),
                    style: ElevatedButton.styleFrom(
                      padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 12),
                      foregroundColor: Theme.of(context).primaryColor
                    ),
                  ),
                ],
            ),
            const SizedBox(height: 10),          
            ValueListenableBuilder<List<OutputResult>>(
              valueListenable: predictions,
              builder: (context, predictionList, _) {
                if (predictionList.isEmpty) {
                  return const Center(child: Text('Insert an image to obtain a prediction'));
                }                
              return Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Center(
                    child: Text('Inference took ${inferenceTime.value}ms'),
                  ),
                  const SizedBox(height: 10.0),
                  Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: predictionList.map((prediction) {
                    return Padding(
                      padding: const EdgeInsets.symmetric(vertical: 4.0),
                      child: Row(
                        children: [
                          Expanded(
                            flex: 5,
                            child: Stack(
                              children: [
                                LinearProgressIndicator(
                                  value: prediction.confidence,
                                  minHeight: 20.0,
                                  backgroundColor: Colors.black38,
                                  valueColor: AlwaysStoppedAnimation<Color>(Theme.of(context).primaryColor),
                                ),
                                Center(
                                  child: Text(
                                    prediction.className.length > 30 ? prediction.className.substring(0, 30) : prediction.className,
                                    style: const TextStyle(
                                      fontSize: 16.0,
                                      color: Colors.white,
                                      fontWeight: FontWeight.bold,
                                    ),
                                  ),
                                ),
                              ],
                            ),
                          ),
                          const SizedBox(width: 8.0),
                        ],
                      ),
                    );
                   }).toList(),
                  )]);
                },
              ),
          ],
        ),
      ),
    );
  }
}
