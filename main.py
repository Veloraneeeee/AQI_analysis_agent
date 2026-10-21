from typing import Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from pydantic import BaseModel, Field

from pprint import pprint
import pandas as pd
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from pathlib import Path

from langchain_core.messages import SystemMessage,HumanMessage, AIMessage

from uuid import uuid4 # 给每个对象生成一个几乎不会重复的 ID
from langchain_core.messages import HumanMessage

# from pandas.core.interchange import column
from typing import TypedDict, Required, Literal
from openpyxl import load_workbook

from langgraph.types import interrupt, Command

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

from analysis_graph import build_analysis_graph

llm = ChatOllama(
    model="qwen3:1.7b", # 回答概念问题、进行普通对话
    temperature=0,
    reasoning=False, # 关闭这个模型的“思考模式”，让它直接生成回答
)

context_llm = ChatOllama( # 理解用户要怎样修改研究条件，输出结构化操作
    model="qwen3:4b",
    temperature=0,
    reasoning=False,
)

class ResearchContext(BaseModel):
    region: str | None = Field(
        default=None,
        description="用户指定的研究地区；未指定时为 None。",
    )

    year: int | None = Field(
        default=None,
        description="用户指定的研究年份；未指定时为 None。",
    )

    event_description: str | None = Field(
        default=None,
        description="用户描述的天气事件，如暴雨。",
    )

    pollutant: str | None = Field(
        default=None,
        description="用户指定的污染物，如臭氧。",
    )

class ResearchFieldChange(BaseModel):
    field: Literal[
        "region",
        "year",
        "event_description",
        "pollutant",
    ] # field 是改哪个条件？

    operation: Literal[
        "set",
        "clear",
        "restore_first",
    ] # operation 是怎么改
    # set：设置新值。
    # clear：取消条件。
    # restore_first：恢复最初指定的值。

    value: str | int | None # value是设置成什么


class ResearchAction(BaseModel):
    changes: list[ResearchFieldChange] # 可能一个问题里多个修改条件

    answer_request: str | None = Field(
        description=(
            "除修改条件外，本轮还需回答的问题或执行的请求。"
            "只修改条件时为 None；"
            "混合请求保留附带问题，不要回答它。"
        ),
    )

class ResearchContextUpdate(BaseModel):
    updates: ResearchContext = Field(
        default_factory=ResearchContext,
        description=(
            "只填写用户本轮明确新增或修改的研究条件。"
            "未修改的字段保持 None。"
        ),
    )

    clear_fields: list[
        Literal["region", "year", "event_description", "pollutant"]
    ] = Field(
        default_factory=list,
        description=(
            "只有用户明确要求取消某个研究条件时，"
            "才填入对应字段名。普通提问返回空列表。"
        ),
    )

    clear_evidence: str | None = Field(
        default=None,
        description=(
            "用户明确要求取消研究条件的原话片段，"
            "必须逐字来自最新用户消息。"
            "没有明确取消要求时为 None。"
            "题外话、先不分析、概念提问都不是取消要求。"
        ),
    )

class AnalysisState(TypedDict, total=False):

    messages: Annotated[list[BaseMessage], add_messages] # 保存用户和助手的对话，聊天记录
    research_context: dict[str, str | int | None] # 把研究条件保存在图的会话状态里，当前生效的研究条件
    research_history: list[dict[str, str | int | None]] # 过去保存过的条件快照

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
    return aqi_changes


def match_and_calculate(state: AnalysisState) -> dict:
    aqi_changes = analyze_event_aqi(
        event_file_path=state["event_file_path"],
        aqi_file_path=state["aqi_file_path"],
        column_mapping=state["column_mapping"],
        select_event_type=state["selected_event_type"],
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



    # 以user说的：当前研究2024年，用户说：“把年份改成2023年，顺便解释一下AQI是什么。” 为例子
def chat_with_user(state: AnalysisState) -> dict:
    #负责处理用户的一轮输入：理解要求、更新研究条件、组织回答，再把结果交回图保存。

    # state AnalysisState 里已经有：
    # - messages：包含本轮用户消息的聊天记录。
    # - research_context：上一轮保存的研究条件。
    # - research_history：过去的条件快照。

    # 这个函数只处理一轮。外面的 while True 负责不断接收下一句话。

    # 读取旧研究条件
    current_context = ResearchContext.model_validate(
        state.get("research_context", {})  # 先取出保存的条件。第一次聊天时还没有，就得到空字典 {}。
    ) # 按照 ResearchContext 的格式进行校验，并生成一个对象。

    # 读到这里的旧年份是 2024。model_validate() 按照我们定义的格式校验数据，
    # 这里的 current_context，后面一直代表本轮修改前的条件。

    context_instruction = SystemMessage(
        content=(
                """
    解析最新用户消息，返回 changes 和 answer_request。

    changes 是研究条件操作列表。
    answer_request 是需要另外回答的问题或处理的请求。

    只修改条件时 answer_request=null。
    既修改又提问时，两者都填写。
    普通问答或闲聊时 changes=[]，
    answer_request 保留用户的问题或消息。

    不要把“先不分析”当作待执行的分析请求。
    没有明确修改要求时，changes=[]。
    假设问题中明确说先别修改，也不修改。

    操作规则：
    set：明确设置新值，value 填新值，年份用整数。
    clear：明确取消条件，value=null。
    restore_first：恢复最初指定值，value=null，由程序查历史。

    field 只能是：
    region：地区
    year：年份
    event_description：天气事件
    pollutant：污染物

    只修改明确要求的字段。
    “先不分析”仍允许保存已指定条件。
    需要完整提取本轮给出的条件，不得遗漏事件或污染物。

    例：研究得州2024年的暴雨与臭氧AQI，先不分析。
    {"changes":[{"field":"region","operation":"set","value":"得州"},{"field":"year","operation":"set","value":2024},{"field":"event_description","operation":"set","value":"暴雨"},{"field":"pollutant","operation":"set","value":"臭氧"}],"answer_request":null}

    例：改成2023年
    {"changes":[{"field":"year","operation":"set","value":2023}],"answer_request":null}

    例：把年份改成2023年，顺便解释一下AQI是什么。
    {"changes":[{"field":"year","operation":"set","value":2023}],"answer_request":"解释一下AQI是什么。"}

    例：取消年份限制，解释一下臭氧是什么。
    {"changes":[{"field":"year","operation":"clear","value":null}],"answer_request":"解释一下臭氧是什么。"}

    例：把年份改回最开始那个年份
    {"changes":[{"field":"year","operation":"restore_first","value":null}],"answer_request":null}

    例：AQI是什么意思？
    {"changes":[],"answer_request":"AQI是什么意思？"}

    例：你妈妈不喜欢你
    {"changes":[],"answer_request":"你妈妈不喜欢你"}
    """
                + "\n当前条件："
                + current_context.model_dump_json()
        )
    )

    context_model = context_llm.with_structured_output( # 调用4b模型，理解本轮要求
        ResearchAction
    )
    action = context_model.invoke(
        [context_instruction, state["messages"][-1]] # 最新用户消息
    )
    # 读到这里期望得到：
    # {
    #   "changes": [
    #     {
    #       "field": "year",
    #       "operation": "set",
    #       "value": 2023
    #     }
    #   ],
    #   "answer_request": "解释一下AQI是什么。"
    # }
    # 这一步只理解要求，还没有实际修改条件。

    # Python 执行条件操作
    merged_context = current_context.model_dump()
    update_notes = [] # 遇到无效值或找不到历史时，把原因放进 update_notes
    # 例子这里最终执行 merged_context["year"] = 2023

    for change in action.changes:  # 根据ResearchFieldChange里面定义的 field_name 修改研究条件字典中的对应项
        # changes对应 class RA
        field_name = change.field

        if change.operation == "set": # 检查新值格式，再更新指定字段
            if change.value is None:
                update_notes.append(
                    f"{field_name} 没有提供新值，保留原条件。"
                )
                continue

            if field_name == "year":
                valid_value = type(change.value) is int
            else:
                valid_value = (
                    isinstance(change.value, str)
                    and change.value.strip() != ""
                )

            if valid_value is False:
                update_notes.append(
                    f"{field_name} 的新值格式无效，保留原条件。"
                )
                continue

            merged_context[field_name] = change.value

        elif change.operation == "clear": # 把指定字段设为 None
            merged_context[field_name] = None

        elif change.operation == "restore_first": # 从历史中查找该字段最初的非空值
            for snapshot in state.get("research_history", []):
                historical_value = snapshot.get(field_name)

                if historical_value is not None:
                    merged_context[field_name] = historical_value
                    break
            else:
                update_notes.append(
                    f"没有找到 {field_name} 最初指定的历史值，"
                    "未执行恢复。"
                )

    # 校验新条件，并记录变化历史
    updated_context = ResearchContext.model_validate(
        merged_context # 本轮合并修改后的字典，再校验一次
    )

    research_history = [
        snapshot.copy()
        for snapshot in state.get("research_history", [])
    ]

    # 得到前后两份快照：
    previous_snapshot = current_context.model_dump()
    # current_context：这一轮开始时读取的研究条件。
    # .model_dump()：把 Pydantic 对象转换成普通 Python 字典。
    # previous_snapshot：保存修改前的那份字典。
    current_snapshot = updated_context.model_dump()
    # updated_context：处理用户本轮要求后得到的条件。
    # current_snapshot：保存修改后的字典。

    # previous_snapshot：年份2024
    # current_snapshot：年份2023

    # 两者不同，就把新快照 2023 加入 research_history

    if current_snapshot != previous_snapshot:
        # 研究条件确实改变了，才记录一条新的历史快照。
        if not research_history and any(
            value is not None
            for value in previous_snapshot.values()
        ):
            research_history.append(previous_snapshot.copy())

        research_history.append(current_snapshot.copy())

    answer_instruction = SystemMessage(
        content=(
            "你是一名面向研究者的天气与空气质量分析助手。"
            "请使用中文自然、简洁地回答，并结合对话理解追问。\n"
            "程序为本轮整理的研究条件是：\n"
            f"{updated_context.model_dump_json()}\n"
            f"本轮操作提示：{update_notes}\n"
            "描述当前研究条件时，以程序提供的条件为准，"
            "不要用聊天历史里的旧值代替。"
            "操作提示说明未执行成功时，不得声称操作已完成。"
            "如果这些条件与用户的明确要求冲突，请指出问题。"
            "关键条件有歧义时，提出简短澄清问题。"
            "当前尚未接入数据查询、计算和联网工具。"
            "可以解释一般概念，但不能声称已经查询数据、"
            "完成计算或获取最新信息，也不能编造分析数字。"
            "不必每轮重复介绍能力限制。"
            "用户询问概念时，直接解释概念，"
            "不要强行联系当前研究，也不要添加未经验证的天气与污染关系。"
            "用户说先不分析时，简短确认保存的条件即可。"
            "项目已有内置历史数据，不要默认要求用户上传或提供数据。"
            "目前工具尚未接入；用户要求实际计算时，"
            "如实说明分析功能仍在接入，不要把功能缺失说成用户缺少数据。只提取最新一条用户消息要求的研究条件变更。"
            "已有条件是当前状态，历史对话只用于理解最新消息中的指代。"
            "例如，用户先指定2024年，后来改为2023年，"
            "最新消息说“改回最开始那个年份”，应该把年份更新为2024。"
            "不要把历史中的要求重新执行。"
            "最新消息只是询问概念时，不修改任何研究条件。"
            "无法确定指代时，不猜测、不修改对应字段。"
            "只有确实缺少完成任务所必需的信息时才追问，"
            "不要每轮都以问题或索要更多信息结尾。"
        )
    )

    # 根据真实变化生成确认文字
    changed_fields = [
        # 找出本轮具体改变了哪些字段
        field_name
        for field_name in current_snapshot
        if current_snapshot[field_name] != previous_snapshot[field_name]
    ]
    # 例子这里得到["year"]

    field_labels = {
        "region": "地区",
        "year": "年份",
        "event_description": "天气事件",
        "pollutant": "污染物",
    }

    # 接着 Python 从 current_snapshot 取出新值，生成：
    reply_parts = [] # 准备一个列表，收集本轮回复内容
    # reply_parts = ["年份已设为：2023。"]

    for field_name in changed_fields:
        value = current_snapshot[field_name]
        label = field_labels[field_name]

        if value is None:
            reply_parts.append(f"{label}限制已取消。")
        else:
            reply_parts.append(f"{label}已设为：{value}。")

    reply_parts.extend(update_notes)

    if action.changes and not changed_fields and not update_notes:
        reply_parts.append(
            "研究条件与当前设置一致，未发生变化。"
        )

    # 有问题时，调用1.7b模型回答
    answer_request = action.answer_request

    if answer_request is not None and answer_request.strip() != "":
        question_instruction = SystemMessage(
            content=(
                    answer_instruction.content
                    + "\n条件操作的确认由程序单独提供。"
                      "本次只回答用户的附带问题，不重复确认条件，"
                      "也不要声称执行了新的条件修改。"
            )
        )

        # 例子的问题这里本轮有附带问题，所以进入：
        answer = llm.invoke(
            [
                question_instruction,
                *state.get("messages", [])[:-1],
                HumanMessage(content=answer_request),
            ]
        )
        # 传给回答模型的内容包括：
        # - 回答规则和更新后的研究条件。
        # - 前面的聊天记录。
        # - 本轮需要回答的“解释一下AQI是什么”。

        reply_parts.append(str(answer.content))
        # 生成的回答追加进去：
        # reply_parts = [
        #     "年份已设为：2023。",
        #     "AQI 是空气质量指数……",
        # ]
        # 只修改条件、没有问题时，这次回答模型调用就可以跳过。因此每轮不一定调用两次模型。

    if not reply_parts:
        answer = llm.invoke(
            [answer_instruction, *state.get("messages", [])] # 取出已有聊天记录，没有时使用空列表。
            # 把列表里的消息逐条展开。 # answer_instruction：本轮回答规则。
        )
        reply_parts.append(str(answer.content))

    # 合成回答，返回本轮状态更新
    response = AIMessage(
        content="\n\n".join(reply_parts)
    )

    return {
        "messages": [response],
        "research_context": current_snapshot,
        "research_history": research_history,
    }
# |     返回内容        |            后续作用 |
# |--------------------|--------------------------------------|
# | `messages`         | 新回答通过 `add_messages` 合并进聊天记录 |
# | `research_context` | 保存本轮最新条件 |
# | `research_history` | 保存更新后的历史快照列表 |



analysis_graph = build_analysis_graph(
    state_schema=AnalysisState,
    nodes={
        "inspect_files": inspect_files,
        "identify_columns": identity_columns,
        "ask_for_confirmation": ask_for_confirmation,
        "revise_columns": revise_columns,
        "extract_event_types": extract_event_types,
        "select_event_type": select_event_type,
        "validate_data": validate_data,
        "explain_missing_data": explain_missing_data,
        "match_and_calculate": match_and_calculate,
        "summarize_results": summarize_results,
        "generate_report": generate_report,
    },
    confirmation_router=route_confirmation,
    validation_router=route_validation,
)



chat_builder = StateGraph(AnalysisState)
chat_builder.add_node("chat_with_user", chat_with_user)

chat_builder.add_edge(START, "chat_with_user")
chat_builder.add_edge("chat_with_user", END)

graph = chat_builder.compile(
    checkpointer=InMemorySaver(),
)

# 第一次invoke
if __name__ == "__main__":
    config = {
        "configurable": {
            "thread_id": str(uuid4()),
        }
    }

    print("天气与空气质量助手已启动。输入“退出”结束。")

    while True:
        user_input = input("\n你：").strip()

        if user_input.lower() in {"退出", "exit", "quit", "0"}:
            print("本次对话结束。")
            break

        if user_input == "":
            continue # 跳过本轮后面的代码，回到循环开头。这样就不会把空消息发给模型。

        result = graph.invoke(
            {
                "messages": [
                    HumanMessage(content=user_input),
                ],
            },
            config=config,
        )

        print("\n助手：", result["messages"][-1].content)
        print("\n当前研究条件：")
        pprint(result.get("research_context", {}))

        print("\n研究条件历史：")
        pprint(result.get("research_history", []))