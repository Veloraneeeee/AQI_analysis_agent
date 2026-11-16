"""上传文件分析：文件预览、列名识别与确认、事件选择及分析报告。

不依赖 main；模型与状态结构由调用方在构建分析图时传入。
实际 AQI 计算仍由 aqi_analysis 提供。图的入口和连线保持原样。
"""
from pathlib import Path
from typing import Literal

import pandas as pd
from openpyxl import load_workbook
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langgraph.types import interrupt

from analysis_graph import build_analysis_graph
from aqi_analysis import analyze_event_aqi, classify_trend


from functools import partial

@tool # 识别文件以及数据
def inspect_table(file_path)->str:
    """
    Users会给你一个或者一些文件，这些文件大概率都是csv和xlsx的文件。
    所以在这里需要你做的是识别，查看这些表格的结构，
    比如表格的第一行，每个attributes的名称，如果是xlsx就还输出这个表格的标题。
    查看清楚以后需要你输出一些行(5行左右)作为sample。
    """

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
    """
    这个工具用于针对读取之后的文件进行提取，Users给你的文件基本都是关于天气、AQI数值、
    或者一些自然洪水下雨等灾害的类型Lists，以及该事件的结束的时间(年份月份)引导的文件。
    在这些文件里，我需要你提取Users指定的与自然灾害事件有关的一个attribute然后输出，
    但是无论这个自然灾害发生了多少次，每一个最后输出的type of the stormevents 只能有一个，
    绝对不可以重复输出两个一样的元素。
    """
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



def inspect_files(state: dict) -> dict[str, str]:
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



def identity_columns(state: dict, *, llm) -> dict:
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



def ask_for_confirmation(state: dict)->dict:
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



def route_confirmation(
        state: dict
)->Literal["extract_event_types","revise_columns"]:
    if state["user_confirmed"]:
        return "extract_event_types"

    return "revise_columns"



def revise_columns(state: dict, *, llm) -> dict:
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



def extract_event_types(
        state: dict
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
        state: dict
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
        state: dict,
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
        state: dict
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
    state: dict
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



def match_and_calculate(state: dict) -> dict:
    aqi_changes = analyze_event_aqi(
        event_file_path=state["event_file_path"],
        aqi_file_path=state["aqi_file_path"],
        column_mapping=state["column_mapping"],
        select_event_type=state["selected_event_type"],
    )

    return {
        "aqi_changes": aqi_changes,
    }



def summarize_results(
        state:dict,
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



def generate_report(state: dict, *, llm) -> dict[str, str]:
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



def build_uploaded_file_analysis_graph(state_schema, llm):
    """绑定调用方提供的模型，创建上传文件分析图。"""
    return build_analysis_graph(
        state_schema=state_schema,
        nodes={
            "inspect_files": inspect_files,
            "identify_columns": partial(identity_columns, llm=llm),
            "ask_for_confirmation": ask_for_confirmation,
            "revise_columns": partial(revise_columns, llm=llm),
            "extract_event_types": extract_event_types,
            "select_event_type": select_event_type,
            "validate_data": validate_data,
            "explain_missing_data": explain_missing_data,
            "match_and_calculate": match_and_calculate,
            "summarize_results": summarize_results,
            "generate_report": partial(generate_report, llm=llm),
        },
        confirmation_router=route_confirmation,
        validation_router=route_validation,
    )
