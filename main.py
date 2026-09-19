import pandas as pd
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from pathlib import Path
from typing_extensions import TypedDict
from openpyxl import load_workbook

llm = ChatOllama(
    model = "qwen3:4b",
    temperature = 0
)

class AnalysisState(TypedDict):
    file_path: str
    event_type_column: str
    selected_types: list[str]

@tool # 识别文件以及数据
def inspect_table(file_path)->str:
    """Users会给你一个或者一些文件，这些文件大概率都是csv和xlsx的文件。
    所以在这里我需要你做的是识别，查看这些表格的结构，
    比如表格的第一行，每个attributes的名称，如果是xlsx就还输出这个表格的标题。
    查看清楚以后需要你输出一些行作为sample。"""
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


@tool # 提取不同的时间
def extract_types(file_path: str, event_type_column: str)->set[str]:
    """这个工具用于针对读取之后的文件进行提取，Users给你的文件基本都是关于天气、AQI数值、
    或者一些自然洪水下雨等灾害的类型Lists，以及该事件的结束的时间(年份月份)引导的文件。
    在这些文件里，我需要你提取Users指定的与自然灾害事件有关的一个attribute然后输出，
    但是无论这个自然灾害发生了多少次，最后输出的type of the stormevents 只有一个。"""
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

        unique_types = sorted(types)
        return {f"Users指定的列是：{event_type_column}；所有的types都是：{unique_types}"}

    except ValueError as error:
        raise ValueError(f"The column of '{event_type_column}' was not found"
                         ) from error

    finally:
        workbook.close()


# def select_event(file_path: str,select_types: str)->list[str]:
