# 用户上传一份天气事件表和一份 AQI 表；Agent 自动识别关键字段，请用户确认；
# 用户选择一种天气事件后，Agent 比较该事件开始当天和结束当天的 AQI，并总结整体升降趋势。
from dataclasses import field

from pprint import pprint
import pandas as pd
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from pathlib import Path

# from pandas.core.interchange import column
from typing import TypedDict, Required, Literal
from openpyxl import load_workbook

from langgraph.types import interrupt, Command

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

llm = ChatOllama(
    model = "qwen3:4b",
    temperature = 0
)

class AnalysisState(TypedDict, total=False):

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
    selected_event_type: str

    # validate_data 产生
    validation_error: str | None
    is_valid: bool
    validation_issue: str | None
    final_report: str

    # match_and_calculate 产生
    aqi_changes: list[dict]

    # summarize_results
    trend_summary: dict
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
        "event_file_preview": event_type_preview,
        "aqi_file_preview": aqi_preview
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
    event_location_type: str | None = Field(
        description=(
            "指示事件地点是县还是其他地理区域的列（例如 CZ_TYPE）"
        )
    )


def identity_columns(state: AnalysisState) -> dict:
    structured_llm = llm.with_structured_output(ColumnMapping)
    prompt = f"""
    你现在在检查两个表。

    确定比较天气事件开始和结束时的 AQI 所需的确切列名。

    【天气事件】的文件路径: {state["event_file_path"]}
    【天气事件】文件预览：  {state["event_file_preview"]}

    【AQI】的文件路径:    {state["aqi_file_path"]}
    【AQI】文件预览：     {state["aqi_file_preview"]}



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
def ask_for_confirmation(state: AnalysisState)->dict:
    user_response = interrupt({
        "message":"请确认系统是否正确识别所需字段？",
        "column_mapping":state["column_mapping"],
        "instruction":(
            "如果正确，请回复正确"
            "如需更改，请说明错误字段，并且告知正确的列名"
        )
    })

    feedback = str(user_response).strip()

    confirmation_response = {
        "正确",
        "确认",
        "对",
        "OK",
        "ok",
        "yes",
        "yep",
        "是",
        "y",
    }

    user_confirmed = (
            feedback.lower() in confirmation_response
    )

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
        "column_mapping": revise_mapping.model_dump(),
        "user_confirmed": False,
    }

# 确认后的，提取的noded
def extract_event_types(
        state: AnalysisState
)->dict[str, set[str]]:
    event_type_column = state["column_mapping"]["event_type"]

    if not isinstance(event_type_column, str):
        raise ValueError(
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
        "selected_event_type": str(user_selection).strip(),
    }


def validate_data(
        state: AnalysisState,
) -> dict:

    try:
        selected_event_type = state["selected_event_type"]
        available_event_types = state["available_event_types"]
        column_mapping = state["column_mapping"]

    except KeyError as error:
        return {
            "is_valid": False,
            "validation_error": (
                f"State 中缺少必要数据：{error}"
            ),
            "validation_issue": "missing_state",
        }

    # 用户输入的事件类型不存在
    if selected_event_type not in available_event_types:
        return {
            "is_valid": False,
            "validation_error": (
                f"找不到事件类型：{selected_event_type}"
            ),
            "validation_issue": "invalid_event_type",
        }

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

    missing_fields = []

    for field_name in required_fields:
        field_value = column_mapping.get(field_name)

        if field_value is None:
            missing_fields.append(field_name)

        elif isinstance(field_value, str):
            if not field_value.strip():
                missing_fields.append(field_name)

        elif isinstance(field_value, list):
            if not field_value:
                missing_fields.append(field_name)

    if missing_fields:
        return {
            "is_valid": False,
            "validation_error": (
                f"缺少必要字段：{missing_fields}"
            ),
            "validation_issue": "missing_columns",
        }

    return {
        "is_valid": True,
        "validation_error": None,
        "validation_issue": None,
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
    select_event_type = state["selected_event_type"]

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

#   用同一个监测站和 _end_date 再匹配一次，取得 end_aqi
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

# 统一州和县代码的格式，
# 州代码宽度 = 2
# 县代码宽度 = 3
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


def summarize_results(
        state:AnalysisState,
)->dict[str, dict]:
    aqi_changes = state["aqi_changes"]
    selected_event_type = state["selected_event_type"]

    # 如果没有有效的事件--监测站匹配结果
    if not aqi_changes:
        return {
            "trend_summary":{
                "event_type": selected_event_type,
                "matched_event_count": 0,
                "monitor_site_count": 0,
                "valid_comparison_count": 0,
                "increased_count": 0,
                "decreased_count": 0,
                "unchanged_count": 0,
                "median_aqi_change": None,
                "overall_trend": "no_data",
            }
        }
    changes_data = pd.DataFrame(aqi_changes)

    # 统计每种方向变化了多少次
    trend_counts = changes_data["trend"].value_counts()

    increased_counts = int(
        trend_counts.get("increased", 0)
    )
    decreased_counts = int(
        trend_counts.get("decreased", 0)
    )
    unchanged_counts = int(
        trend_counts.get("unchanged", 0)
    )

    # 所有有效“事件—监测站”比较的 AQI 变化中位数
    median_aqi_change = float(
        changes_data["aqi_change"].median()
    )
    # 用中位数判断这种天气事件的整体趋势
    overall_trend = classify_trend(
        median_aqi_change
    )

    trend_summary = {
        "event_type": selected_event_type,

        # 实际获得完整开始、结束 AQI 的天气事件数量
        "matched_event_count": int(
            changes_data["event_id"].nunique()
        ),

        # 涉及不同监测站数量
        "monitor_site_count": int(
            changes_data["monitor_site_id"].nunique()
        ),

        # 一条记录代表一个事件和一个监测站比较
        "valid_comparison_count": int(
            len(changes_data)
        ),

        "increased_count": increased_counts,
        "decreased_count": decreased_counts,
        "unchanged_count": unchanged_counts,

        "median_aqi_change": median_aqi_change,
        "overall_trend": overall_trend,
    }

    return {
        "trend_summary": trend_summary,
    }

def generate_report(
        state: AnalysisState,
) -> dict[str, str]:

    trend_summary = state["trend_summary"]

    # 没有有效匹配结果时，不需要调用 LLM
    if trend_summary["overall_trend"] == "no_data":
        report = (
            f"无法分析天气事件 "
            f"{trend_summary['event_type']} 的 AQI 变化。\n"
            "没有找到同时包含事件开始当天和结束当天 AQI 的"
            "有效事件—监测站记录。"
        )

        return {
            "final_report": report,
        }

    prompt = f"""
                 你是一名空气质量数据分析助手。

                 下面是某种天气事件在开始当天和结束当天的 AQI 变化汇总：{trend_summary}

                 请根据这些数据生成一份简洁的中文分析报告。

                 报告需要包含：

                 1. 分析的天气事件类型。
                 2. 成功匹配的天气事件数量。
                 3. 涉及的监测站数量。
                 4. 有效的事件—监测站比较次数。
                 5. AQI 升高、降低和不变的次数。
                 6. AQI 变化值的中位数。
                 7. 根据中位数说明整体 AQI 是升高、降低还是不变。
                 8. 简要说明分析限制：
                     - 事件与 AQI 只按照州、县和日期匹配。
                     - 这里只比较事件开始当天和结束当天。
                     - 结果表示数据中的关联，不能直接证明天气事件导致了 AQI 变化。

                 规则：

                 - 只能使用提供的数据。
                 - 不要编造原因、数字或结论。
                 - AQI 变化值等于结束当天 AQI 减去开始当天 AQI。
                 - 正数代表 AQI 升高，负数代表 AQI 降低。
                 - 使用自然、清楚的中文。
"""

    response = llm.invoke(prompt)

    report = response.content

    if not isinstance(report, str):
        report = str(report)

    return {
        "final_report": report,
    }

# 终于要连图了！！耶耶耶
builder = StateGraph(
    state_schema=AnalysisState,
)
builder.add_node("inspect_files",       inspect_files,)
builder.add_node("identify_columns",    identity_columns,)
builder.add_node("ask_for_confirmation",ask_for_confirmation,)
builder.add_node("revise_columns",      revise_columns,)
builder.add_node("extract_event_types", extract_event_types,)
builder.add_node("select_event_type",   select_event_type,)
builder.add_node("validate_data",       validate_data,)
builder.add_node("explain_missing_data",explain_missing_data,)
builder.add_node("match_and_calculate", match_and_calculate,)
builder.add_node("summarize_results",   summarize_results,)
builder.add_node("generate_report",     generate_report,)

builder.add_edge(
    START,
    "inspect_files")

builder.add_edge(
    "inspect_files",
    "identify_columns")

builder.add_edge(
    "identify_columns",
    "ask_for_confirmation")

# ask_for_confirmation连接第一条条件边
builder.add_conditional_edges(
    "ask_for_confirmation",
            route_confirmation,
                   {
                             "extract_event_types": "extract_event_types",
                             "revise_columns": "revise_columns",
                             },)
builder.add_edge(
    "revise_columns",
    "ask_for_confirmation",)

builder.add_edge(
    "extract_event_types",
    "select_event_type")

builder.add_edge(
    "select_event_type",
    "validate_data")

builder.add_conditional_edges(
    "validate_data",
    route_validation,
    {
        "select_event_type": "select_event_type",
        "explain_missing_data": "explain_missing_data",
        "match_and_calculate": "match_and_calculate",
    },)

builder.add_edge(
    "explain_missing_data",
    END)

builder.add_edge(
    "match_and_calculate",
    "summarize_results",)

builder.add_edge(
    "summarize_results",
    "generate_report",)

builder.add_edge(
    "generate_report",
    END)

checkpointer = InMemorySaver()

graph = builder.compile(
    checkpointer=checkpointer,
)

# 第一次invoke
if __name__ == "__main__":
    config={
        "configurable":{
            "thread_id":"aqi-analysis-1",
        }
    }
    initial_state = {
        "event_file_path": (
            "/Users/velorane/Desktop/"
            "StormEvents_Combined_Fixed.xlsx"
        ),
        "aqi_file_path": (
            "/Users/velorane/26 spring/STAT605-project/"
            "final/data/combined_daily_44201_2023_2025.csv"
        ),
    }

    result = graph.invoke(
        initial_state,
        config=config,
    )

    while "__interrupt__" in result:
        interrupt_request = result["__interrupt__"][0].value

        print("\nAgent 已暂停，请处理下面的请求：")
        pprint(
            interrupt_request,
            width=100,
            sort_dicts=False,
        )

        user_response = input(
            "\n请输入你的回复："
        ).strip()

        result = graph.invoke(
            Command(resume=user_response),
            config=config,
        )

    print("\nAgent 已完成运行。")

    if "final_report" in result:
        print(result["final_report"])
    else:
        print(result)

    if interrupt:
        print("\nAgent 已暂停，请处理下面的请求：")
        print(interrupt[0].value)

    else:
        print(result)