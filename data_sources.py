from pathlib import Path

import pandas as pd


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

# 字典：记录地区叫法与州代码的对应关系
# 把用户的地区名称转换成数据里的代码
REGION_STATE_FIPS = {
    "得州": "48",
    "德州": "48",
    "德克萨斯州": "48",
    "texas": "48",
    "tx": "48",
}

# 函数：去掉首尾空格、统一英文大小写，再查对应关系。无法识别时明确报错，避免意外查询整个数据集。
def resolve_state_fips(region: str) -> str: # 负责查表
    normalized_region = region.strip().casefold()

    state_fips = REGION_STATE_FIPS.get(normalized_region)

    if state_fips is None:
        raise ValueError(
            f"暂不支持地区“{region}”，当前支持得州及其别名。"
        )

    return state_fips

def read_table(
    file_path: str,
    columns: list[str],
) -> pd.DataFrame:

    path = Path(file_path)

    if path.suffix.lower() == ".csv":
        return pd.read_csv(
            path,
            usecols=columns,
        )

    if path.suffix.lower() == ".xlsx":
        return pd.read_excel(
            path,
            usecols=columns,
        )

    raise ValueError(
        f"Unsupported file type: {path.suffix}"
    )


def normalize_fips(
    values: pd.Series,
    width: int,
) -> pd.Series:
    numeric_values = pd.to_numeric(
        values,
        errors="coerce",
    )

    return (
        numeric_values
        .astype("Int64")
        .astype("string")
        .str.zfill(width)
    )


def get_weather_event_types(region: str, year: int) -> dict:
    """
    查询指定州、指定年份的天气事件类型。

    按事件开始日期所在年份筛选。
    返回匹配记录行数和不重复的事件类型。
    当前地区支持范围由 resolve_state_fips 决定。
    """
    state_fips = resolve_state_fips(region)   # 得到 "48"

    if not 1 <= year <= 9999:
        raise ValueError("年份必须在1至9999之间。")

    state_column = COLUMN_MAPPING["event_state"]
    start_column = COLUMN_MAPPING["start_time"]
    type_column = COLUMN_MAPPING["event_type"]

    event_data = pd.read_excel(  # 读取州代码、开始日期、事件类型三列
        EVENT_FILE_PATH,
        usecols=[
            state_column,
            start_column,
            type_column,
        ],
    )

    state_codes = normalize_fips(
        event_data[state_column],
        width=2,
    )

    region_data = event_data.loc[  # 筛选得州记录。
        state_codes == state_fips
    ].copy()

    start_dates = pd.to_datetime(
        region_data[start_column],
        errors="coerce",
    )

    invalid_date_count = int(start_dates.isna().sum())

    if invalid_date_count > 0:
        raise ValueError(
            f"该地区有{invalid_date_count}条记录的开始日期"
            "缺失或无法识别，暂不能可靠地按年份筛选。"
        )

    matched_data = region_data.loc[  # 再筛选开始年份为2023的记录
        start_dates.dt.year == year
    ]

    event_types = sorted(
        {
            str(value).strip()
            for value in matched_data[type_column].dropna()
            if str(value).strip()
        }
    )

    return {
        "region": region,
        "state_fips": state_fips,
        "year": year,
        "year_basis": "事件开始日期所在年份",
        "matched_row_count": len(matched_data),    # 匹配的数据行数
        "event_types": event_types,
        "event_type_count": len(event_types),      # 不重复的天气事件类型数
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