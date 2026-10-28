import pandas as pd

from data_sources import (
    EVENT_FILE_PATH,
    AQI_FILE_PATH,
    COLUMN_MAPPING,
    resolve_state_fips,
)

from data_cache import read_cached_excel
from time import perf_counter

def build_event_windows(
    start_time: str | pd.Timestamp,
    end_time: str | pd.Timestamp,
    before_days: int = 3,
    after_days: int = 3,
) -> dict[str, pd.Timestamp]:

    if (
        type(before_days) is not int
        or type(after_days) is not int
        or before_days < 1
        or after_days < 1
    ):
        raise ValueError("事件前后窗口天数必须是正整数。")

    start_date = pd.Timestamp(start_time)
    end_date = pd.Timestamp(end_time)

    if pd.isna(start_date) or pd.isna(end_date): # 是在检查：start_date 是不是缺失值
        raise ValueError("事件开始和结束时间不能为空。")

    if end_date < start_date:
        raise ValueError("事件结束时间不能早于开始时间。")

    start_date = start_date.normalize()
    end_date = end_date.normalize()

    return {
        "before_start": start_date - pd.Timedelta(days=before_days),
        "before_end": start_date - pd.Timedelta(days=1),
        "after_start": end_date + pd.Timedelta(days=1),
        "after_end": end_date + pd.Timedelta(days=after_days),
    }


def compare_monitor_windows(
    monitor_data: pd.DataFrame,  # 必须只包含一个监测站的数据
    windows: dict[str, pd.Timestamp],
) -> dict:

    dates = pd.to_datetime(
        monitor_data["Date Local"],
        errors="raise",
    ).dt.normalize()

    values = pd.to_numeric(
        monitor_data["AQI"],
        errors="raise",
    )

    daily_data = pd.DataFrame(
        {
            "date": dates,
            "aqi": values,
        }
    ).dropna(subset=["date", "aqi"])

    daily_aqi = daily_data.groupby("date")["aqi"].median()

    before_dates = pd.date_range(
        windows["before_start"],
        windows["before_end"],
        freq="D",
    )

    after_dates = pd.date_range(
        windows["after_start"],
        windows["after_end"],
        freq="D",
    )

    before_values = daily_aqi.reindex(before_dates)
    # 它把数据对齐到窗口要求的每一天。哪天没有记录，那个位置就是缺失值，因此可以检查是否完整。
    after_values = daily_aqi.reindex(after_dates)

    before_complete = bool(before_values.notna().all())
    after_complete = bool(after_values.notna().all())

    before_mean = (
        float(before_values.mean())
        if before_complete
        else None
    )

    after_mean = (
        float(after_values.mean())
        if after_complete
        else None
    )

    aqi_change = (
        after_mean - before_mean
        if before_mean is not None and after_mean is not None
        else None
    )

    return {
        "before_expected_days": len(before_dates),
        "before_observed_days": int(before_values.notna().sum()),
        "after_expected_days": len(after_dates),
        "after_observed_days": int(after_values.notna().sum()),
        "before_mean_aqi": before_mean,
        "after_mean_aqi": after_mean,
        "aqi_change": aqi_change,
    }
    # expected_days：窗口要求有多少天数据。
    # observed_days：实际有有效AQI数据的天数。

def load_selected_events(
    region: str,
    year: int,
    event_type: str,
) -> pd.DataFrame:

    state_fips = resolve_state_fips(region)

    state_column = COLUMN_MAPPING["event_state"]
    type_column = COLUMN_MAPPING["event_type"]
    start_column = COLUMN_MAPPING["start_time"]
    end_column = COLUMN_MAPPING["end_time"]

    event_data = read_cached_excel(
        EVENT_FILE_PATH,
        columns=[
            state_column,
            type_column,
            start_column,
            end_column,
            COLUMN_MAPPING["event_county"],
            COLUMN_MAPPING["event_location_type"],
        ],
    )

    state_codes = pd.to_numeric(
        event_data[state_column],
        errors="raise",
    )

    selected_events = event_data.loc[
        (state_codes == int(state_fips))
        & (
            event_data[type_column]
            .astype("string")
            .str.strip()
            .str.casefold()
            == event_type.strip().casefold()
        )
    ].copy()

    selected_events["_start_date"] = pd.to_datetime(
        selected_events[start_column],
        errors="raise",
    )

    if selected_events["_start_date"].isna().any():
        raise ValueError("所选地区和类型存在缺失的事件开始日期。")

    selected_events = selected_events.loc[
        selected_events["_start_date"].dt.year == year
    ].copy()

    if selected_events.empty:
        raise ValueError(
            f"内置数据中没有匹配记录："
            f"{region}，{year}年，{event_type}。"
        )

    selected_events["_end_date"] = pd.to_datetime(
        selected_events[end_column],
        errors="raise",
    )

    if selected_events["_end_date"].isna().any():
        raise ValueError("筛选出的事件记录存在缺失的结束日期。")

    return selected_events.reset_index(drop=True)



# 这里的监测站标识继续沿用项目已有配置：
# ["State Code", "County Code", "Site Num", "POC"]
def load_region_aqi(region: str) -> pd.DataFrame:
    state_fips = resolve_state_fips(region)

    state_column = COLUMN_MAPPING["monitor_state"]
    county_column = COLUMN_MAPPING["monitor_county"]
    site_columns = COLUMN_MAPPING["monitor_site_id"]

    required_columns = list(
        dict.fromkeys(
            [
                COLUMN_MAPPING["aqi_date"],
                COLUMN_MAPPING["aqi_value"],
                state_column,
                county_column,
                *site_columns,
            ]
        )
    )

    aqi_data = read_cached_excel(
        AQI_FILE_PATH,
        columns=required_columns,
    )

    state_codes = pd.to_numeric(
        aqi_data[state_column],
        errors="raise",
    )

    region_aqi = aqi_data.loc[
        state_codes == int(state_fips)
    ].copy()

    if region_aqi.empty:
        raise ValueError(
            f"内置臭氧AQI数据中没有找到地区：{region}。"
        )

    for column in site_columns:
        codes = pd.to_numeric(
            region_aqi[column],
            errors="raise",
        ).astype("Int64")

        if codes.isna().any():
            raise ValueError(
                f"监测站标识列“{column}”存在缺失值，"
                "无法可靠地区分监测站。"
            )

        region_aqi[column] = codes.astype("string")

    region_aqi["_state_code"] = (
        region_aqi[state_column].str.zfill(2)
    )

    region_aqi["_county_code"] = (
        region_aqi[county_column].str.zfill(3)
    )

    region_aqi["_monitor_site_id"] = (
        region_aqi[site_columns].agg("|".join, axis=1)
    )

    return region_aqi.reset_index(drop=True)



def compare_event_with_monitors(
    event: pd.Series,
    region_aqi: pd.DataFrame,
    before_days: int = 3,
    after_days: int = 3,
) -> dict:

    location_type = str(
        event[COLUMN_MAPPING["event_location_type"]]
    ).strip().upper()

    if location_type != "C":
        raise ValueError(
            "这条事件不是县级记录，不能直接按县匹配监测站。"
        )

    state_value = event[COLUMN_MAPPING["event_state"]]
    county_value = event[COLUMN_MAPPING["event_county"]]

    if pd.isna(state_value) or pd.isna(county_value):
        raise ValueError("事件的州代码或县代码缺失。")

    state_code = f"{int(state_value):02d}"
    county_code = f"{int(county_value):03d}"

    windows = build_event_windows(
        start_time=event["_start_date"],
        end_time=event["_end_date"],
        before_days=before_days,
        after_days=after_days,
    )

    county_aqi = region_aqi.loc[
        (region_aqi["_state_code"] == state_code)
        & (region_aqi["_county_code"] == county_code)
    ] # 州、县必须同时相同，因为不同州可能使用相同的县代码

    monitor_results = []

    for monitor_id, monitor_data in county_aqi.groupby(
        "_monitor_site_id"
    ):
        comparison = compare_monitor_windows(
            monitor_data,
            windows,
        )
        # 把同县数据按监测站拆开。每次传给 compare_monitor_windows() 的数据只属于一个站，确保前后比较不会混用站点。
        monitor_results.append(
            {
                "monitor_site_id": monitor_id,
                **comparison, # **comparison 表示把已有计算结果展开，放进这条监测站结果中。
            }
        )

    complete_count = sum(
        result["aqi_change"] is not None
        for result in monitor_results
    )

    return {
        "state_code": state_code,
        "county_code": county_code,
        "event_start": event["_start_date"].isoformat(),
        "event_end": event["_end_date"].isoformat(),
        "windows": {
            name: value.date().isoformat()
            for name, value in windows.items()
        },
        "matched_monitor_count": len(monitor_results),
        "complete_monitor_count": complete_count,
        "incomplete_monitor_count": (
            len(monitor_results) - complete_count
        ),
        "monitor_results": monitor_results,
    }


def analyze_event_windows(
    region: str,
    year: int,
    event_type: str,
    before_days: int = 3,
    after_days: int = 3,
) -> dict:

    analysis_started = perf_counter()
    print("[分析 1/3] 正在筛选天气事件……", flush=True)

    events = load_selected_events(
        region=region,
        year=year,
        event_type=event_type,
    )

    print(
        f"[分析 1/3] 已筛出{len(events)}条记录，"
        f"累计耗时{perf_counter() - analysis_started:.2f}秒。",
        flush=True,
    )

    print("[分析 2/3] 正在准备臭氧AQI数据……", flush=True)

    region_aqi = load_region_aqi(region)

    county_mask = (
        events[COLUMN_MAPPING["event_location_type"]]
        .astype("string")
        .str.strip()
        .str.upper()
        .eq("C")
        .fillna(False)
    )

    county_events = events.loc[county_mask]

    event_results = []
    event_mean_changes = []

    without_monitor_count = 0
    without_complete_window_count = 0
    complete_pair_count = 0
    incomplete_pair_count = 0

    print(
        f"[分析 3/3] 正在比较{len(county_events)}条县级记录……",
        flush=True,
    )

    for event_row_id, event in county_events.iterrows():
        comparison = compare_event_with_monitors(
            event=event,
            region_aqi=region_aqi,
            before_days=before_days,
            after_days=after_days,
        )

        valid_changes = [
            monitor["aqi_change"]
            for monitor in comparison["monitor_results"]
            if monitor["aqi_change"] is not None
        ]

        event_mean_change = (
            sum(valid_changes) / len(valid_changes)
            if valid_changes
            else None
        )

        if comparison["matched_monitor_count"] == 0:
            without_monitor_count += 1

        elif not valid_changes:
            without_complete_window_count += 1

        if event_mean_change is not None:
            event_mean_changes.append(event_mean_change)

        complete_pair_count += comparison["complete_monitor_count"]
        incomplete_pair_count += comparison["incomplete_monitor_count"]

        event_results.append(
            {
                "event_row_id": int(event_row_id),
                "event_mean_aqi_change": event_mean_change,
                **comparison,
            }
        )

    overall_change = (
        sum(event_mean_changes) / len(event_mean_changes)
        if event_mean_changes
        else None
    )

    print(
        "[分析] 完成，"
        f"可完整比较{len(event_mean_changes)}条记录，"
        f"总耗时{perf_counter() - analysis_started:.2f}秒。",
        flush=True,
    )

    return {
        "region": region,
        "year": year,
        "event_type": event_type,
        "before_days": before_days,
        "after_days": after_days,
        "matched_event_rows": len(events),
        "county_event_rows": len(county_events),
        "excluded_non_county_rows": (   # 非县级记录，当前不能按县匹配
            len(events) - len(county_events)
        ),
        "rows_without_monitor": without_monitor_count, # 县级记录，但内置AQI中没有同县站点
        "rows_without_complete_window": ( # 有同县站点，但没有一个站的前后窗口都完整
            without_complete_window_count
        ),
        "rows_with_complete_comparison": len(event_mean_changes), # 至少有一个站可以完整比较的事件记录数
        "complete_event_monitor_pairs": complete_pair_count,
        "incomplete_event_monitor_pairs": incomplete_pair_count, # 完整的“事件记录＋监测站”组合数
        "mean_event_row_aqi_change": overall_change,
        "event_results": event_results,
    }
# | 字段                             | 含义                     |
# |---------------------------------|-------------------------|
# | `excluded_non_county_rows`      | 非县级记录，当前不能按县匹配 |
# | `rows_without_monitor`          | 县级记录，但内置AQI中没有同县站点 |
# | `rows_without_complete_window`  | 有同县站点，但没有一个站的前后窗口都完整 |
# | `rows_with_complete_comparison` | 至少有一个站可以完整比较的事件记录数 |
# | `complete_event_monitor_pairs`  | 完整的“事件记录＋监测站”组合数 |


def compare_event_years(
    region: str,
    baseline_year: int,
    target_year: int,
    event_type: str,
    before_days: int = 3,
    after_days: int = 3,
) -> dict:
    if any(
        type(year) is not int or not 1 <= year <= 9999
        for year in (baseline_year, target_year)
    ):
        raise ValueError("两个年份都必须是1至9999之间的整数。")

    if baseline_year == target_year:
        raise ValueError("基准年份和目标年份不能相同。")

    if any(
        type(days) is not int or days < 1
        for days in (before_days, after_days)
    ):
        raise ValueError("前后窗口天数必须是正整数。")

    annual_results = {}

    for label, year in (
        ("baseline", baseline_year),
        ("target", target_year),
    ):
        print(f"[年份比较] 正在分析{year}年……", flush=True)

        result = analyze_event_windows(
            region=region,
            year=year,
            event_type=event_type,
            before_days=before_days,
            after_days=after_days,
        )

        annual_results[label] = {
            name: value
            for name, value in result.items()
            if name != "event_results"
        }

    baseline_change = annual_results["baseline"][
        "mean_event_row_aqi_change"
    ]

    target_change = annual_results["target"][
        "mean_event_row_aqi_change"
    ]

    difference = (
        target_change - baseline_change
        if baseline_change is not None and target_change is not None
        else None
    )

    return {
        "region": region,
        "event_type": event_type,
        "baseline_year": baseline_year,
        "target_year": target_year,
        "before_days": before_days,
        "after_days": after_days,
        "baseline": annual_results["baseline"],
        "target": annual_results["target"],
        "change_difference": difference,
        "status": (
            "ok"
            if difference is not None
            else "insufficient_data"
        ),
        "comparison_basis": (
            "目标年份的平均前后变化量"
            "减去基准年份的平均前后变化量"
        ),
    }


if __name__ == "__main__":
    from pprint import pprint

    result = compare_event_years(
        region="得州",
        baseline_year=2023,
        target_year=2024,
        event_type="Flash Flood",
        before_days=7,
        after_days=7,
    )

    print("\n两个年份的比较结果：")
    pprint(result, sort_dicts=False)