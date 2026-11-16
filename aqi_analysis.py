import pandas as pd

from data_sources import (
    EVENT_FILE_PATH,
    AQI_FILE_PATH,
    COLUMN_MAPPING,
    resolve_state_fips,
    read_table,
    normalize_fips,
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

    daily_aqi = daily_data.groupby("date")["aqi"].median() # 同站同日多条记录取 .median()

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
        float(before_values.mean()) # 前、后窗口分别取 .mean()
        if before_complete
        else None
    )

    after_mean = (
        float(after_values.mean()) # 前、后窗口分别取 .mean()
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

    print("[分析 2/3] 正在准备AQI数据……", flush=True)

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
            sum(valid_changes) / len(valid_changes) # 同一事件内，各完整站点变化取平均
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
        sum(event_mean_changes) / len(event_mean_changes) # 同一事件内，各完整站点变化取平均
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



# |           分析步骤        | 事件前后窗口分析                   | 开始日／结束日分析         |
# |--------------------------|---------------------------------|-------------------------|
# | 同一监测站、同一天有多条 AQI | 中位数，合成当天一个值              | 中位数，合成当天一个值     |
# | 每个站的变化量             | 后窗口平均值 − 前窗口平均值          | 结束当天值 − 开始当天值    |
# | 同一事件涉及多个完整站点     | 对站点变化量取平均值                | 保留各站比较记录          |
# | 最终汇总                  | 先得到各事件平均变化，再对事件取平均值 | 对所有事件—站点变化取中位数 |



# 上传文件的开始日/结束日比较；与上面的前后窗口分析保持不同口径。
def analyze_event_aqi(
    event_file_path: str,
    aqi_file_path: str,
    column_mapping: dict[str, str | list[str] | None],
    select_event_type: str,
) -> list[dict]:

    # 列出AQI文件需要读取的字段
    monitor_site_columns = column_mapping["monitor_site_id"]
    if not isinstance(monitor_site_columns, list):
        raise ValueError(
            "monitor_site_id 应包含列的列表。."
        )


    aqi_columns = [
        column_mapping["aqi_date"],
        column_mapping["aqi_value"],
        column_mapping["monitor_state"],
        column_mapping["monitor_county"],
    ] + monitor_site_columns
    aqi_columns = list(dict.fromkeys(aqi_columns)) # monitor_site_id 中可能已经包含州和县字段，需要去重

    aqi_data = read_table(
        aqi_file_path,
        aqi_columns,
    )

    # 读取天气事件表
    event_columns = [
        column_mapping["event_type"],
        column_mapping["start_time"],
        column_mapping["end_time"],
        column_mapping["event_state"],
        column_mapping["event_county"],
        column_mapping["event_location_type"],
    ]

    if not all( # 这些值理论上都应该是列名字符串
        isinstance(column, str)
        for column in event_columns
    ):
        raise ValueError(
            "天气事件文件的必要列没有被正确识别。"
        )

    event_columns = [
        column
        for column in event_columns
        if isinstance(column, str)
    ]

    event_data = read_table( # match_and_calculate 中应该同时存在：event_data，aqi_data
        event_file_path,
        event_columns,
    )

    # 下一步只筛选用户选择的事件类型
    event_type_column = column_mapping["event_type"]

    if not isinstance(event_type_column, str):
        raise ValueError(
            "event_type 没有对应到有效的列名。"
        )

    selected_events = event_data[ # event_data 中筛选用户选择的事件类型。
        event_data[event_type_column] # 取得事件类型整列
        .astype(str)
        .str.strip() # 把值转换为字符串并去掉首尾空格
        == select_event_type
        ].copy()

    if selected_events.empty: # 检查是否真的筛到了数据
        raise ValueError(
            f"没有找到事件类型：{select_event_type}"
        )

#    event_data
#    → 包含全部天气事件
#    selected_events
#    → 只包含用户选中的天气事件


    # 现在统一事件的开始和结束日期。不要覆盖原始列，创建两个内部使用的新列。
    start_time_column = column_mapping["start_time"]
    end_time_column = column_mapping["end_time"]

    if not isinstance(start_time_column, str):
        raise ValueError(
            "start_time 没有对应到有效的列名。"
        )

    if not isinstance(end_time_column, str):
        raise ValueError(
            "end_time 没有对应到有效的列名。"
        )

#   转换日期：
    selected_events["_start_date"] = pd.to_datetime(
        selected_events[start_time_column],
        errors="coerce", # 表示无法识别的日期会变成 NaT，而不是让程序崩溃。
    ).dt.normalize() # 会去掉具体时分秒，只保留当天

    selected_events["_end_date"] = pd.to_datetime(
        selected_events[end_time_column],
        errors="coerce",
    ).dt.normalize()

    selected_events = selected_events.dropna( # 删除无法识别日期的事件
        subset=["_start_date", "_end_date"]
    )

    if selected_events.empty:
        raise ValueError(
            "所选事件没有可用的开始和结束日期。"
        )

#   下一步处理 AQI 表里的日期和 AQI 数值
    event_location_type_column = column_mapping[
        "event_location_type"
    ]

    if not isinstance(event_location_type_column, str):
        raise ValueError(
            "event_location_type 没有对应到有效的列名。"
        )

    # 只保留县级事件
    selected_events = selected_events[
        selected_events[event_location_type_column]
        .astype(str)
        .str.strip()
        .str.upper()
        == "C"
        ].copy()

    # 如果用户选择的事件没有县级记录
    if selected_events.empty:
        raise ValueError(
            "所选事件没有可以按县匹配的记录。"
        )

#   下一步处理 AQI 表里的日期和 AQI 数值，先取得对应列名
    aqi_date_column = column_mapping["aqi_date"]
    aqi_value_column = column_mapping["aqi_value"]

    if not isinstance(aqi_date_column, str):
        raise ValueError(
            "aqi_date 没有对应到有效的列名。"
        )

    if not isinstance(aqi_value_column, str):
        raise ValueError(
            "aqi_value 没有对应到有效的列名。"
        )

#   把日期转换成和事件日期相同的格式
    aqi_data["_aqi_date"] = pd.to_datetime(
        aqi_data[aqi_date_column],
        errors="coerce",
    ).dt.normalize()

#   把 AQI 转换成数字
    aqi_data["_aqi_value"] = pd.to_numeric(
        aqi_data[aqi_value_column],
        errors="coerce",
    )

    aqi_data = aqi_data.dropna(
        subset=["_aqi_date", "_aqi_value"]
    )

    if aqi_data.empty:
        raise ValueError(
            "AQI 文件中没有可用的日期和 AQI 数值。"
        )

    #   在匹配州和县之前，需要补一个字段：天气事件表里的县代码只有在 CZ_TYPE = "C" 时才能当作县代码使用


    # 这一步统一州和县代码的格式，州代码补成两位、县代码补成三位。
    # 取得四个地点列
    event_state_column = column_mapping["event_state"]
    event_county_column = column_mapping["event_county"]
    monitor_state_column = column_mapping["monitor_state"]
    monitor_county_column = column_mapping["monitor_county"]

    # 检查它们都是字符串
    location_columns = [
        event_state_column,
        event_county_column,
        monitor_state_column,
        monitor_county_column,
    ]

    if not all(
            isinstance(column, str)
            for column in location_columns
    ):
        raise ValueError(
            "州或县字段没有被正确识别。"
        )

    # 创建统一后的匹配列
    selected_events["_state_code"] = normalize_fips(
        selected_events[event_state_column],
        width=2,
    )

    selected_events["_county_code"] = normalize_fips(
        selected_events[event_county_column],
        width=3,
    )

    # AQI 表
    aqi_data["_state_code"] = normalize_fips(
        aqi_data[monitor_state_column],
        width=2,
    )

    aqi_data["_county_code"] = normalize_fips(
        aqi_data[monitor_county_column],
        width=3,
    )

    # 最后删除地点代码无法识别的记录
    selected_events = selected_events.dropna(
        subset=["_state_code", "_county_code"]
    )

    aqi_data = aqi_data.dropna(
        subset=["_state_code", "_county_code"]
    )
# 现在两张表都拥有统一的：
    # _state_code
    # _county_code

    # 接下来把 AQI 数据整理成“每个监测站每天一条记录”，避免同一站点同一天有重复行。

    # 1. 创建监测站 ID
    aqi_data["_monitor_site_id"] = (
        aqi_data[monitor_site_columns]
        .astype("string")
        .fillna("")
        .agg("|".join, axis=1)
    )

    # 2. 处理同站同日重复记录
    daily_aqi = (
        aqi_data
        .groupby(
            [
                "_state_code",
                "_county_code",
                "_monitor_site_id",
                "_aqi_date",
            ],
            as_index=False,
        )["_aqi_value"]
        .median() # 这里使用中位数处理同一监测站同一天的重复 AQI 记录。
    )


    # 下一步把selected_events 的 _start_date
    # 与：
    # daily_aqi 的 _aqi_date
    # 按同州、同县匹配，从而得到每个监测站的 start_aqi。
    # 1. 给每场事件一个内部 ID
    selected_events = selected_events.reset_index(drop=True)
    selected_events["_event_row_id"] = selected_events.index

    # 2. 按州、县和开始日期匹配 AQI
    start_matches = selected_events.merge(
        daily_aqi,
        left_on=[
            "_state_code",
            "_county_code",
            "_start_date",
        ],
        right_on=[
            "_state_code",
            "_county_code",
            "_aqi_date",
        ],
        how="inner",
    )

    # 3. 把开始当天的 AQI 改名
    start_matches = start_matches.rename(
        columns={
            "_aqi_value": "start_aqi",
        }
    )

    start_matches = start_matches.drop( # 删除已经完成匹配的 AQI 日期列
        columns=["_aqi_date"]
    )

    if start_matches.empty: # 检查是否存在匹配结果
        raise ValueError(
            "没有找到事件开始当天对应的 AQI 记录。"
        )

    # 用同一个监测站和 _end_date 再匹配一次，取得 end_aqi
    end_aqi_data = daily_aqi.rename( # 1.准备结束日 AQI 数据
        columns={
            "_aqi_date": "_end_aqi_date",
            "_aqi_value": "end_aqi",
        }
    )

    matched_events = start_matches.merge( # 2. 按地点、监测站和结束日期匹配
        end_aqi_data,
        left_on=[
            "_state_code",
            "_county_code",
            "_monitor_site_id",
            "_end_date",
        ],
        right_on=[
            "_state_code",
            "_county_code",
            "_monitor_site_id",
            "_end_aqi_date",
        ],
        how="inner",
    )

    matched_events = matched_events.drop( # 3. 删除辅助日期列
        columns=["_end_aqi_date"]
    )

    if matched_events.empty:
        raise ValueError(
            "没有找到同一监测站在事件开始和结束当天的完整 AQI 记录。"
        )

    matched_events["aqi_change"] = ( # 4. 计算 AQI 变化
            matched_events["end_aqi"]
            - matched_events["start_aqi"]
    )

#   给每条匹配结果标记趋势，并转换成 State 可以保存的记录列表
    # 1.定义升降判断def classify_trend。
    # 2.给每条结果添加趋势
    matched_events["trend"] = (
        matched_events["aqi_change"]
        .apply(classify_trend)
    )
    # 3.创建统一的输出字段
    matched_events["event_type"] = select_event_type

    matched_events["start_date"] = (
        matched_events["_start_date"]
        .dt.strftime("%Y-%m-%d")
    )

    matched_events["end_date"] = (
        matched_events["_end_date"]
        .dt.strftime("%Y-%m-%d")
    )

    # 4. 只保留最终需要的数据
    result_columns = [
        "_event_row_id",
        "event_type",
        "_state_code",
        "_county_code",
        "_monitor_site_id",
        "start_date",
        "end_date",
        "start_aqi",
        "end_aqi",
        "aqi_change",
        "trend",
    ]
         # 转换成 list[dict】
    aqi_changes = (
        matched_events[result_columns]
        .rename(
            columns={
                "_event_row_id": "event_id",
                "_state_code": "state_code",
                "_county_code": "county_code",
                "_monitor_site_id": "monitor_site_id",
            }
        )
        .to_dict(orient="records")
    )

    # 最后保持 node 的返回值：
    return aqi_changes


def classify_trend(aqi_change: float) -> str:
    if aqi_change > 0:
        return "increased"

    if aqi_change < 0:
        return "decreased"

    return "unchanged"



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