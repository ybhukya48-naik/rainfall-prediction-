class RainfallPrediction {
  final String stationName;
  final double probability;
  final double outlook30m;
  final double outlook1h;
  final bool alert;
  final String alertMessage;
  final List<String> safetyTips;
  final List<double> history;
  final Map<String, double> importance;

  RainfallPrediction({
    required this.stationName,
    required this.probability,
    required this.outlook30m,
    required this.outlook1h,
    required this.alert,
    required this.alertMessage,
    required this.safetyTips,
    required this.history,
    required this.importance,
  });

  factory RainfallPrediction.fromJson(Map<String, dynamic> json) {
    return RainfallPrediction(
      stationName: json['station_name'] ?? 'Unknown',
      probability: (json['probability'] as num).toDouble(),
      outlook30m: (json['outlook_30m'] as num).toDouble(),
      outlook1h: (json['outlook_1h'] as num).toDouble(),
      alert: json['alert'] as bool,
      alertMessage: json['alert_message'] ?? '',
      safetyTips: List<String>.from(json['safety_tips'] ?? []),
      history: List<double>.from(json['history'] ?? []),
      importance: Map<String, double>.from(json['importance'] ?? {}),
    );
  }
}
