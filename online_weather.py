from datetime import datetime, timezone
from urllib.parse import urlsplit

import httpx


def get_nws_forecast(latitude: float, longitude: float) -> dict:
    if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
        raise ValueError("经纬度超出有效范围。")

    point_url = (
        f"https://api.weather.gov/points/"
        f"{latitude:.4f},{longitude:.4f}"
    )

    headers = {
        "User-Agent": "AQIResearchAssistant/0.2",
        "Accept": "application/geo+json",
    }

    try:
        with httpx.Client(
            headers=headers,
            timeout=15.0,
        ) as client:
            point_response = client.get(point_url)
            point_response.raise_for_status()

            point_properties = point_response.json()["properties"]
            forecast_url = point_properties.get("forecast")

            if not forecast_url:
                raise ValueError(
                    "NWS没有提供该坐标的常规天气预报。"
                )

            parsed_url = urlsplit(forecast_url)

            if (
                parsed_url.scheme != "https"
                or parsed_url.netloc != "api.weather.gov"
            ):
                raise ValueError("预报链接不是预期的NWS接口。")

            forecast_response = client.get(
                forecast_url,
                params={"units": "si"},
            )
            forecast_response.raise_for_status()

            properties = forecast_response.json()["properties"]

    except httpx.HTTPError as error:
        raise ValueError(
            f"NWS联网请求失败：{error}"
        ) from error

    periods = properties.get("periods") or []

    if not periods:
        raise ValueError("NWS返回的预报时段为空。")

    return {
        "provider": "NOAA / National Weather Service",
        "latitude": latitude,
        "longitude": longitude,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "forecast_updated_at": properties.get("updateTime"),
        "valid_from": periods[0]["startTime"],
        "valid_until": periods[-1]["endTime"],
        "point_source_url": point_url,
        "forecast_source_url": str(forecast_response.url),
        "periods": [
            {
                "name": period["name"],
                "start_time": period["startTime"],
                "end_time": period["endTime"],
                "temperature": period.get("temperature"),
                "temperature_unit": period.get("temperatureUnit"),
                "precipitation_probability": (
                    period.get("probabilityOfPrecipitation") or {}
                ).get("value"),
                "precipitation_probability_unit": (
                    period.get("probabilityOfPrecipitation") or {}
                ).get("unitCode"),
                "short_forecast": period.get("shortForecast"),
            }
            for period in periods
        ],
    }



def search_us_cities(
    city_name: str,
    state_name: str | None = None,
) -> dict:
    city_name = city_name.strip()

    if not city_name:
        raise ValueError("请提供城市名称。")

    state_name = state_name.strip() if state_name else None

    query = (
        f"{city_name}, {state_name}"
        if state_name
        else city_name
    )

    search_limit = 10

    try:
        response = httpx.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={
                "name": query,
                "count": search_limit,
                "language": "en",
                "countryCode": "US",
                "format": "json",
            },
            headers={
                "User-Agent": "AQIResearchAssistant/0.2",
            },
            timeout=15.0,
        )

        response.raise_for_status()
        locations = response.json().get("results", [])

    except httpx.HTTPError as error:
        raise ValueError(
            f"城市查询失败：{error}"
        ) from error

    candidates = [
        {
            "city_id": location["id"],
            "city_name": location["name"],
            "state_name": location.get("admin1"),
            "country_code": location["country_code"],
            "latitude": location["latitude"],
            "longitude": location["longitude"],
            "timezone": location.get("timezone"),
        }
        for location in locations
        if location.get("country_code") == "US"
        and (
            location.get("feature_code") or ""
        ).startswith("PPL")
        and location["name"].strip().casefold() == city_name.casefold()
    ]

    return {
        "provider": "Open-Meteo / GeoNames",
        "query": query,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "source_url": str(response.url),
        "may_be_truncated": len(locations) >= search_limit,
        "candidates": candidates,
    }



def get_city_forecast(
    city_name: str,
    state_name: str | None = None,
    city_id: int | None = None,
) -> dict:
    search = search_us_cities(city_name, state_name)
    candidates = search["candidates"]

    source_info = {
        "city_query": search["query"],
        "geocoding_provider": search["provider"],
        "geocoding_source_url": search["source_url"],
    }

    if not candidates:
        return {
            **source_info,
            "status": "location_not_found",
            "candidates": [],
            "message": (
                "未找到精确匹配的美国城市，"
                "请核对英文城市名和州名。"
            ),
        }

    if city_id is not None:
        location = next(
            (
                item
                for item in candidates
                if item["city_id"] == city_id
            ),
            None,
        )

        if location is None:
            raise ValueError(
                "指定的城市ID不在本次查询候选中，请重新选择。"
            )

    elif len(candidates) == 1 and (
        bool(state_name and state_name.strip())
        or not search["may_be_truncated"]
    ):
        location = candidates[0]

    else:
        return {
            **source_info,
            "status": "choose_location",
            "candidates": candidates,
            "may_be_truncated": search["may_be_truncated"],
            "message": "请补充州名，或选择返回候选中的城市ID。",
        }

    forecast = get_nws_forecast(
        latitude=location["latitude"],
        longitude=location["longitude"],
    )

    return {
        **source_info,
        "status": "ok",
        "location": location,
        "forecast": forecast,
    }



if __name__ == "__main__":
    from pprint import pprint

    result = get_city_forecast(
        city_name="Houston",
        state_name="Texas",
    )

    pprint(
        {
            key: value
            for key, value in result.items()
            if key != "forecast"
        },
        sort_dicts=False,
    )

    if result["status"] == "ok":
        forecast = result["forecast"]

        print("\n预报来源与时间：")
        pprint(
            {
                key: value
                for key, value in forecast.items()
                if key != "periods"
            }
        )

        print("\n前3个预报时段：")
        pprint(forecast["periods"][:3], sort_dicts=False)