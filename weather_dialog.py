from copy import deepcopy
from datetime import datetime, timedelta, timezone
import re
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from langchain_core.tools import BaseTool

from data_sources import REGION_STATE_FIPS


DAY_OFFSETS = {
    "today": 0,
    "tomorrow": 1,
    "day_after_tomorrow": 2,
}


def is_weather_cancellation(question: str, context: dict) -> bool:
    text = question.strip()

    if re.fullmatch(r"取消(?:这次|刚才的)?(?:天气预报|预报|近期天气)查询[。.!！]?", text):
        return True

    return bool(context.get("pending_query")) and bool(
        re.fullmatch(
            r"(?:不查了|算了[，,]?\s*不查了|取消(?:刚才的)?(?:天气)?查询)[。.!！]?",
            text,
        )
    )


def _requested_day(question: str, proposed: str, fallback: str = "available") -> str:
    words = {"今天": "today", "明天": "tomorrow", "后天": "day_after_tomorrow"}
    mentioned = [value for word, value in words.items() if word in question]

    if len(mentioned) == 1:
        return mentioned[0]

    if len(mentioned) > 1 or re.search(r"最近|近期|未来|这周|下周|本周|这个月", question):
        return "available"

    if proposed in DAY_OFFSETS:
        return proposed

    return fallback if fallback in DAY_OFFSETS else "available"


def summarize_weather_forecast(
    location: dict,
    periods: list[dict],
    weather_day: str,
) -> str:
    city = location["city_name"]
    day_label = {
        "today": "今天",
        "tomorrow": "明天",
        "day_after_tomorrow": "后天",
    }.get(weather_day, "在当前可用预报范围内")
    subject = f"{city}{day_label}"

    if not periods:
        return f"{subject}暂无符合条件的预报时段，无法判断是否会下雨。"

    probabilities = [
        period["precipitation_probability"]
        for period in periods
        if isinstance(period.get("precipitation_probability"), (int, float))
        and not isinstance(period["precipitation_probability"], bool)
        and 0 <= period["precipitation_probability"] <= 100
        and period.get("precipitation_probability_unit") == "wmoUnit:percent"
    ]
    sentences = []

    if not probabilities:
        sentences.append(f"{subject}的预报未提供可用降水概率，暂时不能据此判断会不会下雨。")
    elif max(probabilities) == 0:
        sentences.append(f"{subject}已提供的时段降水概率均为0%。")
    else:
        rain_forecast = any(
            re.search(r"rain|showers|drizzle|thunderstorms", str(period.get("short_forecast", "")), re.IGNORECASE)
            and isinstance(period.get("precipitation_probability"), (int, float))
            and not isinstance(period["precipitation_probability"], bool)
            and 0 < period["precipitation_probability"] <= 100
            and period.get("precipitation_probability_unit") == "wmoUnit:percent"
            for period in periods
        )
        precipitation = "降雨" if rain_forecast else "降水"
        if len(probabilities) == len(periods) and len(set(probabilities)) == 1:
            probability_text = f"所列各时段降水概率均为{probabilities[0]:g}%"
        else:
            probability_text = f"已提供的单个时段降水概率最高为{max(probabilities):g}%"
        sentences.append(f"{subject}有{precipitation}可能，{probability_text}。")

    if probabilities and len(probabilities) < len(periods):
        sentences.append("部分时段未提供可用降水概率。")

    temperatures = [
        period["temperature"]
        for period in periods
        if isinstance(period.get("temperature"), (int, float))
        and not isinstance(period["temperature"], bool)
        and period.get("temperature_unit") in {"C", "F"}
    ]
    units = {period.get("temperature_unit") for period in periods}
    if len(temperatures) == len(periods) and len(units) == 1:
        unit = next(iter(units))
        low, high = min(temperatures), max(temperatures)
        temperature_text = f"{low:g}°{unit}" if low == high else f"{low:g}–{high:g}°{unit}"
        sentences.append(f"所列预报时段的气温为{temperature_text}。")

    return "".join(sentences)


def format_weather_forecast(
    result: dict,
    weather_day: str,
    now: datetime | None = None,
) -> str:
    location = result["location"]
    forecast = result["forecast"]
    current_time = now or datetime.now(timezone.utc)

    try:
        city_timezone = ZoneInfo(location["timezone"])
    except (KeyError, TypeError, ZoneInfoNotFoundError) as error:
        raise ValueError("城市时区缺失或无法识别，无法可靠地确定当地日期。") from error

    valid_until = datetime.fromisoformat(forecast["valid_until"])

    if valid_until <= current_time:
        raise ValueError("NWS返回的预报已过期，本次不能作为当前预报使用。")

    periods = [
        period
        for period in forecast["periods"]
        if datetime.fromisoformat(period["end_time"]) > current_time
    ]

    lines = [
        f"天气地点：{location['city_name']}, {location['state_name']}"
        "（城市代表点附近）。",
        f"城市时区：{location['timezone']}。",
        f"查询时间（UTC）：{forecast['fetched_at']}。",
        f"NWS预报更新时间：{forecast['forecast_updated_at']}。",
        f"接口实际覆盖：{forecast['valid_from']} 至 {forecast['valid_until']}。",
    ]

    if weather_day in DAY_OFFSETS:
        target_date = current_time.astimezone(city_timezone).date() + timedelta(
            days=DAY_OFFSETS[weather_day]
        )
        periods = [
            period
            for period in periods
            if datetime.fromisoformat(period["start_time"])
            .astimezone(city_timezone).date() == target_date
        ]
        lines.append(
            f"以下为当地{target_date}开始的NWS预报时段；夜间时段可能延续到次日。"
        )
    elif weather_day != "available":
        raise ValueError("暂不支持这个天气日期选项。")
    else:
        lines.append("以下列出当前可用时段，不代表接口范围之外的整月天气。")

    summary = summarize_weather_forecast(location, periods, weather_day)
    lines = [summary, "", "详细预报与来源：", *lines]

    if not periods:
        lines.append("接口当前没有符合该日期条件的预报时段。")
    else:
        probabilities = [
            period["precipitation_probability"]
            for period in periods
            if period.get("precipitation_probability") is not None
            and period.get("precipitation_probability_unit") == "wmoUnit:percent"
        ]

        if probabilities:
            lines.append(
                f"已提供的单个时段降水概率最高为{max(probabilities)}%；"
                "这些概率不能相加，也不是整个查询范围的累计概率。"
            )
        else:
            lines.append("这些时段没有可用的百分比降水概率，不能将缺失值当成0%。")

        for period in periods:
            start = datetime.fromisoformat(period["start_time"]).astimezone(city_timezone)
            end = datetime.fromisoformat(period["end_time"]).astimezone(city_timezone)
            probability = period.get("precipitation_probability")
            probability_unit = period.get("precipitation_probability_unit")
            probability_text = (
                f"{probability}%"
                if probability is not None and probability_unit == "wmoUnit:percent"
                else "未提供可用百分比"
            )
            temperature = period.get("temperature")
            unit = period.get("temperature_unit")
            temperature_text = (
                f"{temperature}°{unit}"
                if temperature is not None and unit in {"C", "F"}
                else "未提供可用温度"
            )
            lines.append(
                f"- {start:%Y-%m-%d %H:%M} 至 {end:%Y-%m-%d %H:%M}："
                f"气温{temperature_text}，降水概率{probability_text}。"
                f"NWS原文：{period.get('short_forecast') or '未提供'}"
            )

    weather_page_url = (
        "https://forecast.weather.gov/MapClick.php"
        f"?lat={location['latitude']}&lon={location['longitude']}"
    )

    lines.extend(
        [
            f"天气预报网页：{weather_page_url}",
            f"原始预报数据（JSON）：{forecast['forecast_source_url']}",
            "地点数据来源：Open-Meteo / GeoNames。",
        ]
    )
    return "\n".join(lines)


def _candidate_reply(memory: dict) -> str:
    pending = memory.get("pending_query") or {}
    candidates = memory.get("candidates", [])
    lines = [f"请确认天气地点：{pending.get('city_name', '待确定城市')}。"]

    for index, city in enumerate(candidates, start=1):
        lines.append(
            f"{index}. {city['city_name']}, {city.get('state_name')}，"
            f"美国（城市ID：{city['city_id']}）"
        )

    lines.append("请补充州名，或回复“选第1项”等明确选项。")

    if memory.get("geocoding_source_url"):
        lines.append(f"[地点查询来源]({memory['geocoding_source_url']})")

    return "\n".join(lines)


def run_weather_request(
    question: str,
    parameters: dict,
    context: dict,
    forecast_tool: BaseTool,
) -> dict:
    memory = deepcopy(context)
    pending = memory.get("pending_query") or {}
    candidates = memory.get("candidates", [])
    location = memory.get("location") or {}

    def reply(text: str) -> dict:
        return {"reply": text, "weather_context": memory}

    if is_weather_cancellation(question, memory):
        memory.update({"pending_query": None, "pending_status": None, "candidates": []})
        return reply("待完成的天气预报查询已取消，已保存的天气地点保留。")

    city_name = parameters.get("weather_city")
    original = parameters.get("weather_city_original")
    state_name = parameters.get("weather_state")
    city_id = parameters.get("weather_city_id")
    fallback_day = pending.get("weather_day", memory.get("weather_day", "available"))
    weather_day = _requested_day(
        question, parameters.get("weather_day", "available"), fallback_day
    )

    def need_city(text: str) -> dict:
        memory.update(
            {
                "pending_query": {
                    "city_name": None,
                    "state_name": state_name,
                    "city_id": None,
                    "weather_day": weather_day,
                },
                "pending_status": "needs_city",
                "candidates": [],
            }
        )
        return reply(text)

    selection = re.fullmatch(
        r"(?:选|选择|确认)?第?(\d+|[一二两三四五六七八九十])(?:项|个)?[。.!！]?",
        question.strip(),
    )

    if selection:
        numerals = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
                    "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
        token = selection.group(1)
        index = int(token) if token.isdigit() else numerals[token]

        if not pending or not 1 <= index <= len(candidates):
            return reply("当前没有这个城市候选选项，请重新指定城市和州。")

        city_id = candidates[index - 1]["city_id"]

    elif original:
        city_id = None

    if city_id is not None:
        selected_city = next(
            (city for city in candidates if city["city_id"] == city_id),
            None,
        )

        if not pending or selected_city is None:
            return reply("城市ID不在当前待选候选中，请重新指定城市。")

        city_name = selected_city["city_name"]
        state_name = selected_city.get("state_name")

    elif original:
        if original not in question or not city_name:
            return reply("城市名称提取未能与本轮原话对应，请明确提供城市名称。")

        if (
            original.strip().casefold() in REGION_STATE_FIPS
            or original.strip().endswith("州")
            or state_name and city_name.strip().casefold() == state_name.strip().casefold()
        ):
            return need_city("当前工具查询城市附近的预报，请补充具体城市。")

        if not state_name and memory.get("pending_status") == "needs_city":
            state_name = pending.get("state_name")

        if not state_name and city_name.casefold() == str(location.get("city_name", "")).casefold():
            state_name = location.get("state_name")

    elif pending:
        city_name = pending["city_name"]

        if not city_name:
            state_name = state_name or pending.get("state_name")
            return need_city("还缺少具体城市，请补充城市名称。")

        if state_name:
            city_id = None
        elif memory.get("pending_status") == "request_failed":
            state_name = pending.get("state_name")
            city_id = pending.get("city_id")
        elif candidates:
            return reply(_candidate_reply(memory))
        else:
            return reply("上次地点未能匹配，请重新提供英文城市名和州名。")

    elif state_name:
        return need_city("目前只确定了州，请补充具体城市；不会将整州当作单个城市查询。")

    elif location:
        if city_name and city_name.casefold() != location["city_name"].casefold():
            return reply("尚未确认本轮的新城市名称，请明确提供城市。")

        city_name = location["city_name"]
        state_name = location.get("state_name")
        city_id = location["city_id"]

    else:
        return need_city("请提供美国城市和州，例如“休斯敦，得州”。")

    memory["pending_query"] = {
        "city_name": city_name,
        "state_name": state_name,
        "city_id": city_id,
        "weather_day": weather_day,
    }
    memory["pending_status"] = "request_failed"

    if city_id is None:
        memory["candidates"] = []

    print(f"\n[天气] 正在查询{city_name}, {state_name or '待确定州'}的近期预报……", flush=True)

    try:
        result = forecast_tool.invoke(
            {"city_name": city_name, "state_name": state_name, "city_id": city_id}
        )
        memory["geocoding_source_url"] = result["geocoding_source_url"]

        if result["status"] == "choose_location":
            memory["candidates"] = result["candidates"]
            memory["pending_status"] = "choose_location"
            return reply(_candidate_reply(memory))

        if result["status"] == "location_not_found":
            memory["candidates"] = []
            memory["pending_status"] = "location_not_found"
            return reply(result["message"])

        if result["status"] != "ok":
            raise ValueError("天气工具返回了无法识别的状态。")

        answer = format_weather_forecast(result, weather_day)

    except (ValueError, KeyError, TypeError) as error:
        return reply(f"本次天气查询未完成：{error}。")

    memory.update(
        {
            "location": result["location"],
            "candidates": [],
            "pending_query": None,
            "pending_status": None,
            "weather_day": weather_day,
            "last_forecast": result,
        }
    )
    return reply(answer)
