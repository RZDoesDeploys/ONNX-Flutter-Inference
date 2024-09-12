class OutputResult {
  final int id;
  final String className;
  final double confidence;

  OutputResult({
    required this.id,
    required this.className,
    required this.confidence,
  });

  @override
  String toString() {
    return 'OutputResult(id: $id, className: $className, confidence: $confidence)';
  }
}
