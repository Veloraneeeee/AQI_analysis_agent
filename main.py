# 用户上传一份天气事件表和一份 AQI 表；Agent 自动识别关键字段，请用户确认；
# 用户选择一种天气事件后，Agent 比较该事件开始当天和结束当天的 AQI，并总结整体升降趋势。

import pandas as pd
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from pathlib import Path
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
    column_mapping: dict[str, str]
    user_confirmed: bool
    confirmation_feedback: str

    # extract_event_types 和 select_event_type 产生
    available_event_types: set[str]
    selected_event_type: str

    # validate_data 产生
    validation_error: str | None
    is_valid: bool

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
            "如需更改，请说明错误字段，并且告知争取的列名"
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

# 需要修改，revise_columns边
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

# 确认后的，提取的
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
        "available_event_types": available_event_types.model_dump(),
    }