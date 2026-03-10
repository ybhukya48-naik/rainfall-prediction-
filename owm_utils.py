import requests
import os
from dotenv import load_dotenv

load_dotenv()

def get_live_weather(lat, lon):
    """Fetch live weather data from OpenWeatherMap."""
    api_key = os.getenv("OWM_API_KEY")
    if not api_key:
        return {"success": False, "error": "No API Key found"}
    
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return {
                "success": True,
                "temp": data['main']['temp'],
                "hum": data['main']['humidity'],
                "press": data['main']['pressure'],
                "wind": data['wind']['speed'],
                "desc": data['weather'][0]['description'],
                "icon": data['weather'][0]['icon'],
                "city": data['name']
            }
        return {"success": False, "error": f"HTTP {response.status_code}"}
    except Exception as e:
        return {"success": False, "error": str(e)}

def get_live_forecast(lat, lon):
    """Fetch 5-day forecast data from OpenWeatherMap."""
    api_key = os.getenv("OWM_API_KEY")
    if not api_key:
        return {"success": False, "error": "No API Key found"}
    
    try:
        url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={api_key}&units=metric"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return {"success": True, "data": response.json()}
        return {"success": False, "error": f"HTTP {response.status_code}"}
    except Exception as e:
        return {"success": False, "error": str(e)}
