from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"

EVENT_FILE_PATH = DATA_DIR / "StormEvents_Combined_Fixed.xlsx"
AQI_FILE_PATH = DATA_DIR / "combined_daily_44201_2023_2025.xlsx"

COLUMN_MAPPING = {
    "event_type": "EVENT_TYPE",
    "start_time": "BEGIN_DATE_TIME",
    "end_time": "END_DATE_TIME",
    "event_state": "STATE_FIPS",
    "event_county": "CZ_FIPS",
    "event_location_type": "CZ_TYPE",
    "aqi_date": "Date Local",
    "aqi_value": "AQI",
    "monitor_state": "State Code",
    "monitor_county": "County Code",
    "monitor_site_id": [
        "State Code",
        "County Code",
        "Site Num",
        "POC",
    ],
}

if __name__ == "__main__":
    import pandas as pd

    event_columns = pd.read_excel(
        EVENT_FILE_PATH,
        nrows=0,
    ).columns

    aqi_columns = pd.read_excel(
        AQI_FILE_PATH,
        nrows=0,
    ).columns

    required_event_columns = [
        COLUMN_MAPPING["event_type"],
        COLUMN_MAPPING["start_time"],
        COLUMN_MAPPING["end_time"],
        COLUMN_MAPPING["event_state"],
        COLUMN_MAPPING["event_county"],
        COLUMN_MAPPING["event_location_type"],
    ]

    required_aqi_columns = [
        COLUMN_MAPPING["aqi_date"],
        COLUMN_MAPPING["aqi_value"],
        COLUMN_MAPPING["monitor_state"],
        COLUMN_MAPPING["monitor_county"],
        *COLUMN_MAPPING["monitor_site_id"],
    ]

    missing_event_columns = [
        column
        for column in required_event_columns
        if column not in event_columns
    ]

    missing_aqi_columns = [
        column
        for column in required_aqi_columns
        if column not in aqi_columns
    ]

    print("天气事件表缺少的列：", missing_event_columns)
    print("臭氧数据表缺少的列：", missing_aqi_columns)