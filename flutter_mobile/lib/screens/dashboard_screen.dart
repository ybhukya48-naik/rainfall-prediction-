import 'package:flutter/material.dart';
import '../services/weather_service.dart';
import '../models/rainfall_prediction.dart';
import 'dart:async';
import 'dart:math';

class DashboardScreen extends StatefulWidget {
  const DashboardScreen({super.key});

  @override
  _DashboardScreenState createState() => _DashboardScreenState();
}

class _DashboardScreenState extends State<DashboardScreen> {
  final WeatherService _weatherService = WeatherService();
  RainfallPrediction? _currentPrediction;
  String _selectedStation = 'STN_001';
  bool _isLoading = true;
  Timer? _timer;

  @override
  void initState() {
    super.initState();
    _fetchData();
    // Auto-refresh every 10 seconds
    _timer = Timer.periodic(Duration(seconds: 10), (timer) => _fetchData());
  }

  @override
  void dispose() {
    _timer?.cancel();
    super.dispose();
  }

  Future<void> _fetchData() async {
    try {
      final random = Random();
      final seed = _selectedStation == 'STN_001' ? 0 : (_selectedStation == 'STN_002' ? 5 : -5);
      
      final sensorData = {
        'temp': (random.nextDouble() * 20 + 15 + seed),
        'hum': (random.nextDouble() * 40 + 55 + seed),
        'press': (random.nextDouble() * 30 + 995),
        'wind': (random.nextDouble() * 40),
        'btemp': (random.nextDouble() * 60 + 220),
        'reflect': random.nextDouble(),
        'radar': (random.nextDouble() * 60 + seed),
      };

      final prediction = await _weatherService.getPrediction(
        stationId: _selectedStation,
        sensorData: sensorData,
      );

      setState(() {
        _currentPrediction = prediction;
        _isLoading = false;
      });
    } catch (e) {
      print('Error: $e');
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Color(0xFF121212),
      appBar: AppBar(
        backgroundColor: Colors.transparent,
        elevation: 0,
        title: Text('RainAlert IoT', style: TextStyle(fontWeight: FontWeight.bold)),
        actions: [
          DropdownButton<String>(
            dropdownColor: Color(0xFF1E1E1E),
            value: _selectedStation,
            items: [
              DropdownMenuItem(value: 'STN_001', child: Text('Central Hub', style: TextStyle(color: Colors.white))),
              DropdownMenuItem(value: 'STN_002', child: Text('North Ridge', style: TextStyle(color: Colors.white))),
              DropdownMenuItem(value: 'STN_003', child: Text('West Valley', style: TextStyle(color: Colors.white))),
            ],
            onChanged: (val) {
              setState(() {
                _selectedStation = val!;
                _isLoading = true;
              });
              _fetchData();
            },
          ),
          IconButton(icon: Icon(Icons.settings), onPressed: () {}),
        ],
      ),
      body: _isLoading 
        ? Center(child: CircularProgressIndicator())
        : SingleChildScrollView(
            padding: EdgeInsets.all(16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                if (_currentPrediction!.alert)
                  Container(
                    padding: EdgeInsets.all(12),
                    margin: EdgeInsets.only(bottom: 16),
                    decoration: BoxDecoration(
                      color: _currentPrediction!.probability > 0.8 ? Colors.red.withOpacity(0.2) : Colors.orange.withOpacity(0.2),
                      borderRadius: BorderRadius.circular(12),
                      border: Border.all(color: _currentPrediction!.probability > 0.8 ? Colors.red : Colors.orange),
                    ),
                    child: Column(
                      children: [
                        Text(_currentPrediction!.alertMessage, style: TextStyle(color: _currentPrediction!.probability > 0.8 ? Colors.red : Colors.orange, fontWeight: FontWeight.bold)),
                        ..._currentPrediction!.safetyTips.map((tip) => Text('• $tip', style: TextStyle(color: Colors.white70, fontSize: 12))),
                      ],
                    ),
                  ),
                
                // Forecast Card
                Container(
                  padding: EdgeInsets.all(20),
                  decoration: BoxDecoration(
                    gradient: LinearGradient(colors: [Color(0xFF007AFF), Color(0xFF00C7BE)]),
                    borderRadius: BorderRadius.circular(20),
                  ),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text('PREDICTIVE OUTLOOK', style: TextStyle(color: Colors.white70, fontSize: 10, letterSpacing: 1.2)),
                      SizedBox(height: 10),
                      Row(
                        mainAxisAlignment: MainAxisAlignment.spaceAround,
                        children: [
                          _buildOutlookItem('30m', _currentPrediction!.outlook30m),
                          _buildOutlookItem('1h', _currentPrediction!.outlook1h),
                        ],
                      ),
                    ],
                  ),
                ),
                SizedBox(height: 16),
                
                // Main Prob Circle
                Container(
                  padding: EdgeInsets.all(24),
                  decoration: BoxDecoration(color: Color(0xFF1E1E1E), borderRadius: BorderRadius.circular(20)),
                  child: Row(
                    children: [
                      Stack(
                        alignment: Alignment.center,
                        children: [
                          SizedBox(width: 100, height: 100, child: CircularProgressIndicator(value: _currentPrediction!.probability, strokeWidth: 8, color: Color(0xFF007AFF))),
                          Text('${(_currentPrediction!.probability * 100).toInt()}%', style: TextStyle(fontSize: 24, fontWeight: FontWeight.bold, color: Colors.white)),
                        ],
                      ),
                      SizedBox(width: 24),
                      Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(_currentPrediction!.probability > 0.5 ? 'RAIN EXPECTED' : 'CLEAR SKIES', style: TextStyle(fontSize: 18, color: Color(0xFF007AFF), fontWeight: FontWeight.bold)),
                          Text('Last Update: Just now', style: TextStyle(fontSize: 12, color: Colors.grey)),
                        ],
                      )
                    ],
                  ),
                ),
              ],
            ),
          ),
    );
  }

  Widget _buildOutlookItem(String label, double prob) {
    return Column(
      children: [
        Text(label, style: TextStyle(color: Colors.white, fontSize: 12)),
        Text('${(prob * 100).toInt()}%', style: TextStyle(color: Colors.white, fontSize: 24, fontWeight: FontWeight.bold)),
      ],
    );
  }
}
