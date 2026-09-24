# 用户上传一份天气事件表和一份 AQI 表；Agent 自动识别关键字段，请用户确认；
# 用户选择一种天气事件后，Agent 比较该事件开始当天和结束当天的 AQI，并总结整体升降趋势。
from dataclasses import field

import pandas as pd
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from pathlib import Path

from pandas.core.interchange import column
from typing_extensions import TypedDict, Required
from openpyxl import load_workbook

llm = ChatOllama(
    model = "qwen3:4b",
    temperature = 0
)

class AnalysisState(TypedDict):
    # 用户启用Agent时必须提供
    event_file_path: Required[str]
    aqi_file_path: Required[str]

    # inspect_files所产生的
    event_file_preview: str
    aqi_file_preview: str

    # identity_column 和用户确认所产生的
    column_mapping: dict[str, str | list[str] | None]
    user_confirmed: bool
    confirmation_feedback: str

    # extract_event_types 和 select_event_type 产生
    available_event_types: set[str]
    select_event_type: str

    # validate_data 产生
    validation_error: str | None
    is_valid: bool
    validation_issue: str | None
    final_report: str

@tool # 识别文件以及数据
def inspect_table(file_path)->str:
    """Users会给你一个或者一些文件，这些文件大概率都是csv和xlsx的文件。
    所以在这里需要你做的是识别，查看这些表格的结构，
    比如表格的第一行，每个attributes的名称，如果是xlsx就还输出这个表格的标题。
    查看清楚以后需要你输出一些行(5行左右)作为sample。"""
    path = Path(file_path)

    if path.suffix.lower() == ".csv":
        preview = pd.read_csv(path, nrows = 5)
    elif path.suffix.lower() == ".xlsx":
        preview = pd.read_excel(path, nrows = 5)
    else:
        raise (
            ValueError(f"This is not the supported file type: {path.suffix}")
        )

    if path.suffix.lower() == ".xlsx":
        sheet_title = pd.ExcelFile(path).sheet_names[0]
    else:
        sheet_title = None

    sample = preview.to_string(index=False) # DataFrame变成字符串，然后没有索引
    result = f"Sample of the file:\n{sample}\n\n"
    result += f"Attributes: {', '.join(map(str, preview.columns))}" # map把index变成str

    if sheet_title:
        result += f"\nSheet title: {sheet_title}"

    return result


@tool # 提取不同的event_types
def extract_types(file_path: str, event_type_column: str)->set[str]:
    """这个工具用于针对读取之后的文件进行提取，Users给你的文件基本都是关于天气、AQI数值、
    或者一些自然洪水下雨等灾害的类型Lists，以及该事件的结束的时间(年份月份)引导的文件。
    在这些文件里，我需要你提取Users指定的与自然灾害事件有关的一个attribute然后输出，
    但是无论这个自然灾害发生了多少次，每一个最后输出的type of the stormevents 只能有一个，
    绝对不可以重复输出两个一样的元素。"""
    workbook = load_workbook(file_path, read_only=True)

    try:
        rows = workbook.active.iter_rows(values_only=True) # iter_rows作为迭代,从第一行往下一行行取
        headers = next(rows) # 第一行作为表头，迭代器会继续停在第二行。
        column_index = headers.index(event_type_column) # 找目标行在第几列

        types = set() # 集合自动去重

        for row in rows: # 遍历目标list，column_index
            if row[column_index] is not None:
                value = str(row[column_index])
                value = value.strip()
                types.add(value)

        return types

    except ValueError as error:
        raise ValueError(f"The column of '{event_type_column}' was not found"
                         ) from error

    finally:
        workbook.close()


def inspect_files(state: AnalysisState) -> dict[str, str]:
    event_type_preview = inspect_table.invoke({
        "file_path":state["event_file_path"]
    })
    aqi_preview = inspect_table.invoke({
        "file_path":state["aqi_file_path"]
    })
    return {
        "event_type_preview": event_type_preview,
        "aqi_preview": aqi_preview
    }


from pydantic import BaseModel, Field

# 要让LLM查看之前的event_type_preview和aqi_preview，先定义LLM 的结构化输出
class ColumnMapping(BaseModel):
    event_type: str | None = Field(
        description="表示天气事件的类型的一列"
    )
    start_time: str | None = Field(
        description="代表单独的天气事件的开始的当天的一列"
    )
    end_time: str | None = Field(
        description="代表单独的天气事件的结束的当天的一列"
    )
    event_state: str | None = Field(
        description="天气事件文件中的“州”列"
    )
    event_county: str | None = Field(
        description="天气事件文件中的县（County）列"
    )
    aqi_date: str | None = Field(
        description="AQI 文件中的观测日期列"
    )
    aqi_value: str | None = Field(
        description="AQI 文件中的 AQI 值列"
    )
    monitor_state: str | None = Field(
        description="AQI 文件中的“州”列"
    )
    monitor_county: str | None = Field(
        description="AQI 文件中的县（区）列"
    )
    monitor_site_id: list[str] = Field(
        description="共同标识 AQI 监测站点的列"
    )


def identity_columns(state: AnalysisState) -> dict:
    structured_llm = llm.with_structured_output(ColumnMapping)
    prompt = f"""
    你现在在检查两个表。

    确定比较天气事件开始和结束时的 AQI 所需的确切列名。

    【天气事件】的文件路径:{state["event_file_path"]}

    【AQI】的文件路径:    {state["aqi_file_path"]}

    规则：
    - 返回预览中显示的精确列名。
    - 请勿自行编造列名。
    - 若无法识别所需的列，请使用 None。
    - monitor_site_id 可能包含多个列。
    """
    mapping = structured_llm.invoke(prompt)
    return {
        "column_mapping": mapping.model_dump(), # 把普通字段换成字典
    }


# 暂停工作流，展示LLM识别出的字段，并且让用户确认或者纠正
from langgraph.types import interrupt
def ask_for_confirmation(state: AnalysisState)->dict:
    user_response = interrupt({
        "message":"请确认系统是否正确识别所需字段？",
        "column_mapping":state["column_mapping"],
        "instruction":(
            "如果正确，请回复正确"
            "如需更改，请说明错误字段，并且告知正确的列名"
        )
    })

    feedback = {
        "user_response":user_response.strip()
    }

    confirmation_response = {
        "确认"
        "对"
        "OK"
        "ok"
        "yes"
        "yep"
        "是"
    }

    user_confirmed = feedback in confirmation_response

    return {
        "user_confirmed":user_confirmed,
        "confirmation_feedback":feedback,
    }


# 条件边
from typing import Literal

def route_confirmation(
        state: AnalysisState
)->Literal["extract_event_types","revise_columns"]:
    if state["user_confirmed"]:
        return "extract_event_types"

    return "revise_columns"

# 需要修改的：revise_columns边
def revise_columns(state: AnalysisState)->dict:
    structured_llm = llm.with_structured_output(ColumnMapping)

    prompt = f"""
    根据用户的更正，修改当前的列映射。
    
    现在的mapping：{state["column_mapping"]}
    用户要修改的指令：{state["confirmation_feedback"]}
    Weather event preview: {state['event_file_preview']}
    AQI file preview: {state['aqi_file_preview']}
    
    规则：
    - 返回完整的修订后映射。
    - 保留用户未修改的字段。
    - 使用文件预览中的确切列名。
    - 请勿自行编造列名。
    - 无法识别必填列时，请使用 None。
"""
    revise_mapping = structured_llm.invoke(prompt)
    return {
        "revise_columns": revise_mapping.model_dump(),
        "user_confirmed": False,
    }

# 确认后的，提取的noded
def extract_event_types(
        state: AnalysisState
)->dict[str, set[str]] | None:
    event_type_column = state["column_mapping"]["event_type"]

    if not isinstance(event_type_column, str):
        return ValueError(
            "The event type column was not identified correctly."
        )

    available_event_types = extract_types.invoke({
        "file_path": state["event_file_path"],
        "event_type_column": event_type_column,
    })
    return {
        "available_event_types": available_event_types,
    }

def select_event_type(
        state: AnalysisState
) -> dict[str, str]:
    available_event_types = sorted(state["available_event_types"]) # 提取集合

    user_selection = interrupt({
        "message":"请选择一种天气",
        "available_event_types":available_event_types,
    })
    return {
        "select_event_type": str(user_selection).strip(),
    }


def validate_data(
        state: AnalysisState
)->dict:
    # 用户选择的事件类型是否存在于可选集合中。
    select_event_type = state["select_event_type"]
    available_event_types = state["available_event_types"]
    column_mapping = state["column_mapping"]

    try:
        if select_event_type not in available_event_types:
            raise ValueError(
                f'找不到事件类型 f{select_event_type}'
            )

        # LLM 是否识别出了分析需要的所有字段
        required_fields = [
                "event_type",
                "start_time",
                "end_time",
                "event_state",
                "event_county",
                "aqi_date",
                "aqi_value",
                "monitor_state",
                "monitor_county",
                "monitor_site_id",
                "event_location_type",
            ]

        # 找出没有被成功识别的字段
        missing_fields = []
        for i in required_fields:
            if i not in column_mapping.get(i):
                missing_fields.append(i)

        if missing_fields:
            raise ValueError(
                 f"缺少必要字段 :{missing_fields}"
            )

        return {
            "is_valid": True,
            "validation_error": None,
            "validation_issue": None,
        }

    except (KeyError, ValueError) as error: # KeyError对应这个node最开始的selected_event_type etc.
        return {
            "is_valid": False,
            "validation_error": str(error),
            "validation_issue": "missing_columns"
        }

def route_validation( # 条件边不更新节点，只是返回下一站的名字
        state: AnalysisState
)->Literal[
    "select_event_type",
    "explain_missing_data",
    "match_and_calculate",
]:
    if state["is_valid"]:
        return "match_and_calculate"

    if state["validation_issue"]=="invalid_event_type":
        return "select_event_type"

    return "explain_missing_data"

def explain_missing_data(
    state: AnalysisState
)-> dict[str, str]:
    error_message = state["validation_error"]
    report = (
        "目前无法继续分析。\n"
        f"原因：{error_message}\n"
        "请检查上传的文件是否包含需要的字段"
    )
    return {
        "final_report": report,
    }

def match_and_calculate(state: AnalysisState)->dict:
    event_file_path = state["event_file_path"]
    aqi_file_path = state["aqi_file_path"]
    column_mapping = state["column_mapping"]
    select_event_type = state["select_event_type"]

    aqi_changes = list[dict]

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
    aqi_columns = list(dict.fromkeys(aqi_columns))

    aqi_data = read_table(
        aqi_file_path,
        aqi_columns,
    )

    event_columns = [
        column_mapping["event_type"],
        column_mapping["start_time"],
        column_mapping["end_time"],
        column_mapping["event_state"],
        column_mapping["event_county"],
        column_mapping["event_location_type"],
    ]

    if not all(
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

    event_data = read_table(
        event_file_path,
        event_columns,
    )

    event_type_column = column_mapping["event_type"]

    if not isinstance(event_type_column, str):
        raise ValueError(
            "event_type 没有对应到有效的列名。"
        )

    selected_events = event_data[
        event_data[event_type_column]
        .astype(str)
        .str.strip()
        == select_event_type
        ].copy()

    if selected_events.empty:
        raise ValueError(
            f"没有找到事件类型：{select_event_type}"
        )


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

    selected_events["_start_date"] = pd.to_datetime(
        selected_events[start_time_column],
        errors="coerce",
    ).dt.normalize()

    selected_events["_end_date"] = pd.to_datetime(
        selected_events[end_time_column],
        errors="coerce",
    ).dt.normalize()

    selected_events = selected_events.dropna(
        subset=["_start_date", "_end_date"]
    )

    if selected_events.empty:
        raise ValueError(
            "所选事件没有可用的开始和结束日期。"
        )

    event_location_type_column = column_mapping[
        "event_location_type"
    ]

    if not isinstance(event_location_type_column, str):
        raise ValueError(
            "event_location_type 没有对应到有效的列名。"
        )

    selected_events = selected_events[
        selected_events[event_location_type_column]
        .astype(str)
        .str.strip()
        .str.upper()
        == "C"
        ].copy()

    if selected_events.empty:
        raise ValueError(
            "所选事件没有可以按县匹配的记录。"
        )

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

    aqi_data["_aqi_date"] = pd.to_datetime(
        aqi_data[aqi_date_column],
        errors="coerce",
    ).dt.normalize()

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

    event_location_type: str | None = Field(
        description=(
            "Column indicating whether the event location is a county "
            "or another geographic zone, such as CZ_TYPE"
        )
    )

    event_state_column = column_mapping["event_state"]
    event_county_column = column_mapping["event_county"]
    monitor_state_column = column_mapping["monitor_state"]
    monitor_county_column = column_mapping["monitor_county"]

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

    selected_events["_state_code"] = normalize_fips(
        selected_events[event_state_column],
        width=2,
    )

    selected_events["_county_code"] = normalize_fips(
        selected_events[event_county_column],
        width=3,
    )

    aqi_data["_state_code"] = normalize_fips(
        aqi_data[monitor_state_column],
        width=2,
    )

    aqi_data["_county_code"] = normalize_fips(
        aqi_data[monitor_county_column],
        width=3,
    )

    selected_events = selected_events.dropna(
        subset=["_state_code", "_county_code"]
    )

    aqi_data = aqi_data.dropna(
        subset=["_state_code", "_county_code"]
    )

    aqi_data["_monitor_site_id"] = (
        aqi_data[monitor_site_columns]
        .astype("string")
        .fillna("")
        .agg("|".join, axis=1)
    )

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
        .median()
    )

    selected_events = selected_events.reset_index(drop=True)
    selected_events["_event_row_id"] = selected_events.index

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

    start_matches = start_matches.rename(
        columns={
            "_aqi_value": "start_aqi",
        }
    )

    start_matches = start_matches.drop(
        columns=["_aqi_date"]
    )

    if start_matches.empty:
        raise ValueError(
            "没有找到事件开始当天对应的 AQI 记录。"
        )

    end_aqi_data = daily_aqi.rename(
        columns={
            "_aqi_date": "_end_aqi_date",
            "_aqi_value": "end_aqi",
        }
    )

    matched_events = start_matches.merge(
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

    matched_events = matched_events.drop(
        columns=["_end_aqi_date"]
    )

    if matched_events.empty:
        raise ValueError(
            "没有找到同一监测站在事件开始和结束当天的完整 AQI 记录。"
        )

    matched_events["aqi_change"] = (
            matched_events["end_aqi"]
            - matched_events["start_aqi"]
    )

    # 给每条结果添加趋势
    matched_events["trend"] = (
        matched_events["aqi_change"]
        .apply(classify_trend)
    )
    #创建统一的输出字段
    matched_events["event_type"] = select_event_type

    matched_events["start_date"] = (
        matched_events["_start_date"]
        .dt.strftime("%Y-%m-%d")
    )

    matched_events["end_date"] = (
        matched_events["_end_date"]
        .dt.strftime("%Y-%m-%d")
    )

    # 只保留最终需要的数据
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

    return {
        "aqi_changes": aqi_changes,
    }

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

# 统一州和县代码的格式
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

# 定义升降判断
def classify_trend(aqi_change: float) -> str:
    if aqi_change > 0:
        return "increased"

    if aqi_change < 0:
        return "decreased"

    return "unchanged"