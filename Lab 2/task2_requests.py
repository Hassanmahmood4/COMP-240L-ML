"""
Lab Task 2:
Install requests with PIP. Make a GET request to a public API, print the response,
and handle exceptions gracefully.
"""

import requests


def fetch_weather(city: str = "London") -> None:
    """Fetch current weather from Open-Meteo (no API key required)."""
    # Geocoding + weather via Open-Meteo free public APIs
    geo_url = "https://geocoding-api.open-meteo.com/v1/search"
    weather_url = "https://api.open-meteo.com/v1/forecast"

    try:
        geo_response = requests.get(geo_url, params={"name": city, "count": 1}, timeout=10)
        geo_response.raise_for_status()
        geo_data = geo_response.json()

        if not geo_data.get("results"):
            print(f"City '{city}' not found.")
            return

        location = geo_data["results"][0]
        lat = location["latitude"]
        lon = location["longitude"]
        place = location.get("name", city)
        country = location.get("country", "")

        weather_response = requests.get(
            weather_url,
            params={
                "latitude": lat,
                "longitude": lon,
                "current_weather": True,
            },
            timeout=10,
        )
        weather_response.raise_for_status()

        print("Status Code:", weather_response.status_code)
        print("Content-Type:", weather_response.headers.get("Content-Type"))
        print(f"\nWeather for {place}, {country}:")
        print(weather_response.text)

        data = weather_response.json()
        current = data.get("current_weather", {})
        if current:
            print("\n--- Parsed Summary ---")
            print(f"Temperature: {current.get('temperature')} °C")
            print(f"Wind Speed:  {current.get('windspeed')} km/h")
            print(f"Wind Dir:    {current.get('winddirection')}°")
            print(f"Time:        {current.get('time')}")

    except requests.exceptions.Timeout:
        print("Error: The request timed out. Please try again later.")
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the API. Check your internet connection.")
    except requests.exceptions.HTTPError as exc:
        print(f"Error: HTTP error occurred — {exc}")
    except requests.exceptions.RequestException as exc:
        print(f"Error: Request failed — {exc}")
    except ValueError as exc:
        print(f"Error: Failed to parse JSON response — {exc}")


def main():
    print("Making GET request to Open-Meteo public weather API...\n")
    fetch_weather("Islamabad")


if __name__ == "__main__":
    main()
