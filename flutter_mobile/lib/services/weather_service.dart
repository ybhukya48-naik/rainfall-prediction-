import 'dart:convert';
import 'package:http/http.dart' as http;
import '../models/rainfall_prediction.dart';

class WeatherService {
  final String baseUrl = "http://127.0.0.1:5003/api"; // Mapped to backend port

  Future<RainfallPrediction> getPrediction({
    required String stationId,
    double warnThresh = 0.6,
    double critThresh = 0.8,
    required Map<String, dynamic> sensorData,
  }) async {
    final response = await http.post(
      Uri.parse('$baseUrl/predict'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({
        'station_id': stationId,
        'warn_thresh': warnThresh,
        'crit_thresh': critThresh,
        ...sensorData,
      }),
    );

    if (response.statusCode == 200) {
      return RainfallPrediction.fromJson(jsonDecode(response.body));
    } else {
      throw Exception('Failed to load prediction: ${response.body}');
    }
  }

  Future<List<Map<String, dynamic>>> getHistory() async {
    final response = await http.get(Uri.parse('$baseUrl/history'));
    if (response.statusCode == 200) {
      return List<Map<String, dynamic>>.from(jsonDecode(response.body));
    } else {
      throw Exception('Failed to load history');
    }
  }

  Future<Map<String, dynamic>> getLatestStations() async {
    final response = await http.get(Uri.parse('$baseUrl/latest_stations'));
    if (response.statusCode == 200) {
      return jsonDecode(response.body);
    } else {
      throw Exception('Failed to load latest stations');
    }
  }
}
