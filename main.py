from typing import Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from pydantic import BaseModel, Field

from pprint import pprint
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

from langchain_core.messages import SystemMessage,HumanMessage, AIMessage

from uuid import uuid4 # 给每个对象生成一个几乎不会重复的 ID
from langchain_core.messages import HumanMessage

# from pandas.core.interchange import column
from typing import TypedDict, Required, Literal


from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

from uploaded_file_analysis import build_uploaded_file_analysis_graph, extract_types

from data_sources import (
    EVENT_FILE_PATH,
    COLUMN_MAPPING,
    get_weather_event_types,
)

from aqi_analysis import (
    analyze_event_windows,
    compare_event_years,
)

import re

from session_storage import open_session_checkpointer

from operator import add
from analysis_records import make_analysis_record

import json

from saved_results import (
    read_saved_analysis,
    saved_result_headers,
)

from online_weather import get_city_forecast

from weather_dialog import (
    run_weather_request,
    is_weather_cancellation,
)

from contextlib import ExitStack


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

class AnalysisWindow(BaseModel):
    before_days: int = Field(default=3, ge=1)
    after_days: int = Field(default=3, ge=1)


def update_analysis_window(
    question: str,
    current_window: AnalysisWindow,
) -> AnalysisWindow:

    updates = {}

    common = re.search(r"前后\s*(\d+)\s*天", question)
    before = re.search(r"前\s*(\d+)\s*天", question)
    after = re.search(r"后\s*(\d+)\s*天", question)

    if common:
        updates["before_days"] = int(common.group(1))
        updates["after_days"] = int(common.group(1))

    if before:
        updates["before_days"] = int(before.group(1))

    if after:
        updates["after_days"] = int(after.group(1))

    if not updates:
        raise ValueError(
            "请用数字指定窗口，例如前后7天或前5天、后3天。"
        )

    if any(value < 1 for value in updates.values()):
        raise ValueError(
            "窗口天数必须是正整数，原窗口保持不变。"
        )

    return AnalysisWindow.model_validate(
        {**current_window.model_dump(), **updates}
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


class ToolDecision(BaseModel):
    # ToolDecision 定义“选择结果长什么样”，`choose_tool()` 负责根据用户问题，让模型填出这个结果。
    reason: str = Field(
        description=(
            "简短说明选择该工具或不使用工具的原因。"
        ),
    )
    # | 选择                    | 用途           |
    # |---                     |---------------|
    # | `list_weather_events`  | 查询全局事件类型 |
    # | `query_weather_events` | 查询指定地区、年份的事件类型 |
    # | `none`                 | 普通问答或暂不支持的请求 |
    tool_name: Literal[
        "list_weather_events",
        "query_weather_events",
        "analyze_weather_aqi",
        "compare_weather_aqi_years",
        "read_saved_analysis",
        "get_weather_forecast",
        "none",
    ] = Field(
        description=(
            "全局类型查询选择 list_weather_events；"
            "地区年份类型查询选择 query_weather_events；"
            "单年事件前后AQI分析选择 analyze_weather_aqi；"
            "两个年份的事件前后AQI比较选择 compare_weather_aqi_years；"
            "读取保存结果选择 read_saved_analysis；"
            "近期城市天气查询选择 get_weather_forecast；"
            "普通问答或不支持的请求选择 none。"
        ),
    )

    event_names: list[str] = Field(
        description=(
            "用户要求检查的具体事件名称。"
            "保留用户原文，不翻译、不猜测对应类型。"
            "询问全部类型或不需要工具时，返回空列表。"
        )
    )

    # query_region、query_year 是本次查询的参数，与保存研究条件是两个不同的动作。
    query_region: str | None = Field(
        description=(
            "地区年份查询使用的地区。"
            "优先使用本轮问题明确指定的地区；"
            "本轮未指定时使用当前研究地区。"
            "仍无法确定或选择其他工具时，返回 null。"
        )
    )

    query_year: int | None = Field(
        description=(
            "地区年份查询使用的年份。"
            "优先使用本轮问题明确指定的年份；"
            "本轮未指定时使用当前研究年份。"
            "仍无法确定或选择其他工具时，返回 null。"
        )
    )

    query_event_name: str | None = Field(
        default=None,
        description=(
            "AQI分析使用的单个天气事件名称。"
            "优先使用本轮明确指定的原文名称；"
            "未指定时继承当前研究条件的 event_description。"
            "不自行翻译；无法确定或选择其他工具时返回 null。"
        ),
    )

    query_pollutant: str | None = Field(
        default=None,
        description=(
            "AQI分析的污染物。"
            "优先使用本轮明确指定的污染物，"
            "否则继承当前研究条件的 pollutant。"
            "无法确定或选择其他工具时返回 null。"
        ),
    )

    baseline_year: int | None = Field(
        default=None,
        ge=1,
        le=9999,
        description=(
            "跨年比较的基准年份，即被比较的参照年份。"
            "只在选择 compare_weather_aqi_years 时填写。"
            "无法确定时返回 null。"
        ),
    )

    target_year: int | None = Field(
        default=None,
        ge=1,
        le=9999,
        description=(
            "跨年比较的目标年份。"
            "年份间差值为目标年变化量减去基准年变化量。"
            "只在选择 compare_weather_aqi_years 时填写。"
            "无法确定时返回 null。"
        ),
    )

    saved_record_id: str | None = Field(
        default=None,
        description=(
            "需要读取的已保存记录ID。"
            "刚才或最近一次为null；"
            "明确指定旧记录时，从保存目录选择对应ID。"
        ),
    )

    saved_result_view: Literal["summary", "coverage"] = Field(
        default="summary",
        description=(
            "回顾数值用summary；"
            "询问完整记录数量或筛选原因用coverage。"
        ),
    )

    weather_city_original: str | None = Field(
        default=None,
        description=(
            "本轮明确城市名称的原文片段，必须逐字来自问题；"
            "仅有州名或沿用地点时为null。"
        ),
    )

    weather_city: str | None = Field(
        default=None,
        description=(
            "本轮明确城市的英文检索名，"
            "如休斯敦对应Houston；没有新城市时为null。"
        ),
    )

    weather_state: str | None = Field(
        default=None,
        description=(
            "天气查询使用的英文州名，如Texas；"
            "无法确定时为null。"
        ),
    )

    weather_city_id: int | None = Field(
        default=None,
        ge=1,
        description=(
            "用户选择的城市候选ID，"
            "只能从已保存天气候选中选取；其他情况为null。"
        ),
    )

    weather_day: Literal[
        "available",
        "today",
        "tomorrow",
        "day_after_tomorrow",
    ] = Field(
        default="available",
        description=(
            "近期或未指定为available；"
            "今天today，明天tomorrow，后天day_after_tomorrow。"
        ),
    )

class ResearchAction(BaseModel):
    changes: list[ResearchFieldChange] # 可能一个问题里多个修改条件

    answer_request: str | None = Field(
        description=(
            "除修改条件外，本轮还需回答的问题或执行的请求。"
            "只修改条件时为 None；"
            "混合请求保留附带问题，不要回答它。"
        ),
    )

    pending_query_action: Literal[
        "keep",
        "resume",
        "cancel",
    ] = Field(
        default="keep",
        description=(
            "对待完成查询的操作。"
            "用户补充所需条件并要求查询，或明确要求继续时，"
            "返回 resume；"
            "明确取消查询时返回 cancel；"
            "普通问答、仅修改条件、没有待完成查询时返回 keep。"
        ),
    )

    window_action: Literal["keep", "update", "reset"] = Field(
        default="keep",
        description=(
            "明确修改窗口时为update，"
            "明确恢复默认窗口时为reset，其他情况为keep。"
            "概念提问和假设讨论不修改窗口。"
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
    pending_weather_query: str | None  # 保存等待补充参数的原始查询

    analysis_window: dict[str, int]

    analysis_results: Annotated[list[dict], add]

    weather_context: dict

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


# list_weather_events
#     ↓ 使用内置路径和字段配置
# extract_types
#     ↓ 读取真实文件
# 返回排序后的事件类型列表

@tool
def list_weather_events() -> list[str]:
    """
    列出项目内置天气事件数据中的全部事件类型。

    用于回答有哪些天气事件、支持查询哪些事件类型等问题。
    返回整个内置数据集的类型列表，不按当前地区或年份筛选。
    """
    event_types = extract_types.invoke(
        {
            "file_path": str(EVENT_FILE_PATH), # 使用项目内部配置的文件，用户不用提供路径。
            "event_type_column": COLUMN_MAPPING["event_type"], # 找到实际的事件类型列名
        }
    )

    return sorted(event_types) # 把提取出的不重复类型排序，返回列表


@tool
def query_weather_events(region: str, year: int) -> dict:
    """查询指定州、指定年份的天气事件类型。

    按事件开始日期所在年份筛选。
    返回匹配记录行数和不重复的事件类型。
    当前地区支持范围由 resolve_state_fips 决定。
    """
    return get_weather_event_types(region=region, year=year)

@tool
def analyze_weather_aqi(
    region: str,
    year: int,
    event_type: str,
    before_days: int = 3,
    after_days: int = 3,
) -> dict:
    """
    分析指定州、年份和天气类型记录前后的臭氧AQI变化。

    event_type 使用数据中的英文类型名称。
    比较同县、同一监测站的事件前后窗口。
    仅使用前后窗口每日数据均完整的监测站。
    返回描述性变化，不代表因果影响。
    """
    result = analyze_event_windows(
        region=region,
        year=year,
        event_type=event_type,
        before_days=before_days,
        after_days=after_days,
    )

    summary = {
        name: value
        for name, value in result.items()
        if name != "event_results"
    }

    if result["rows_with_complete_comparison"] == 0:
        summary["status"] = "no_complete_comparison"
        summary["message"] = (
            "没有满足同县、同监测站且前后窗口完整的比较记录。"
            "不能计算平均AQI变化，也不能将缺失结果解释为没有变化。"
        )
    else:
        summary["status"] = "ok"
        summary["message"] = (
            "变化量为事件后平均AQI减去事件前平均AQI；"
            "先在每条事件记录内对完整站点取平均，"
            "再对有效事件记录取平均。"
            "结果仅描述前后变化，不代表因果影响。"
        )

    return summary


@tool
def compare_weather_aqi_years(
    region: str,
    baseline_year: int,
    target_year: int,
    event_type: str,
    before_days: int = 3,
    after_days: int = 3,
) -> dict:
    """比较同一州、同一天气类型在两个年份的臭氧AQI前后变化。

    两年使用相同的前后窗口。
    event_type 使用数据中的英文类型名称。
    年份间差值为目标年平均前后变化量减去基准年平均前后变化量。
    """
    result = compare_event_years(
        region=region,
        baseline_year=baseline_year,
        target_year=target_year,
        event_type=event_type,
        before_days=before_days,
        after_days=after_days,
    )

    if result["status"] == "ok":
        message = (
            "两年分别使用各自具备完整窗口的事件记录。"
            "差值比较的是事件前后变化量，"
            "不是全年平均AQI，也不代表因果影响或统计显著性。"
        )
    else:
        message = (
            "至少一个年份没有可用的完整窗口比较结果，"
            "无法计算年份间差值，不能将缺失结果当作0。"
        )

    return {
        **result,
        "message": message,
    }


@tool
def get_weather_forecast(
    city_name: str,
    state_name: str | None = None,
    city_id: int | None = None,
) -> dict:
    """查询美国城市的近期NWS天气预报。

    city_name使用英文城市检索名。
    state_name可提供英文州名以缩小范围。
    city_id只能使用地点查询返回的候选ID。
    返回预报或需要用户补充的地点候选。
    """
    return get_city_forecast(
        city_name=city_name,
        state_name=state_name,
        city_id=city_id,
    )




EVENT_NAME_ALIASES = {
    "龙卷风": "Tornado",
    "暴雨": "Heavy Rain",
    "强降雨": "Heavy Rain",
    "冰雹": "Hail",
    "野火": "Wildfire",
    "森林火灾": "Wildfire",
    "干旱": "Drought",
    "暴雪": "Heavy Snow",
    "雷暴大风": "Thunderstorm Wind",
}

# def choose_tool(question: str：先把当前研究条件、已有分析结果的摘要整理成 JSON，供后面的模型判断应该选哪个工具。
# 收到问题
#     ↓
# 4b模型判断是否需要全局事件类型列表
#     ↓
# 返回选择理由和工具名称

def choose_tool(
    question: str,
    research_context: dict,
    analysis_results: list[dict] | None = None,
    weather_context: dict | None = None,
) -> ToolDecision:

    context_json = ResearchContext.model_validate( # 1. 按 ResearchContext 的定义校验字典，生成模型对象
        research_context
    ).model_dump_json() # 2. 把模型对象转换成 JSON 字符串

    records_json = json.dumps(
        saved_result_headers(analysis_results or []),
        ensure_ascii=False, # 让中文保持可读
    )


    weather_memory = weather_context or {}

    weather_json = json.dumps(
        {
            "location": weather_memory.get("location"),
            "candidates": weather_memory.get("candidates", []),
            "pending_query": weather_memory.get("pending_query"),
        },
        ensure_ascii=False,
    )


    instruction = SystemMessage(
        content=(
            """
你负责选择工具并提取查询参数，不回答问题、不修改研究条件。

工具说明：

1. list_weather_events
查询整个内置数据集的全部事件类型。
适用于明确询问内置数据整体包含哪些类型，
或内置数据整体是否包含某个类型。
这类全局查询不使用当前研究地区和年份。

2. query_weather_events
查询指定地区、指定年份的事件类型。
适用于询问某地某年的事件，
或使用“当地”“这一年”“当前研究范围”等指代。
仅支持州级地区和单个年份，
不支持城市、月份、日期区间、事件次数或AQI计算。

查询参数规则：
- 问题明确指定的地区或年份优先。
- 缺少的参数可以从当前研究条件继承。
- 仍无法确定的参数返回 null，不猜测。
- 即使参数不完整，仍可选择 query_weather_events，
  程序随后会检查缺少什么。
  
当前研究年份为 null，且问题没有明确指定年份时，
query_year 必须为 null，禁止使用示例中的年份。

例：当前研究地区为得州，年份为 null。
问题：当地有哪些天气事件类型？
工具：query_weather_events
query_region="得州"
query_year=null

3. analyze_weather_aqi
分析指定州、单个年份、单个天气类型前后的臭氧AQI变化。
按事件开始年份筛选事件。
仅比较同县、同一监测站的前后窗口。
目前只支持臭氧，不支持其他污染物、综合AQI或臭氧层分析。
本工具仅处理单年分析。
两个年份的比较应选择 compare_weather_aqi_years。
不支持预测、因果判断或多个事件类型同时分析。

参数规则：
query_region、query_year 优先取本轮明确参数，
未提供时继承当前研究条件；仍缺失则返回 null。
query_event_name 优先取本轮明确事件名称，
否则继承 event_description，保留名称原文，不自行翻译。
query_pollutant 优先取本轮明确污染物，
否则继承 pollutant；无法确定则返回 null。
分析请求选择此工具，即使必要参数尚未补齐。
窗口天数由程序解析并从会话状态读取，不需要在工具选择结果中填写。
此工具的 event_names=[]。

只有用户要求实际分析或计算时才选择此工具。
“先不分析”“只记住条件”不能触发分析。

4. compare_weather_aqi_years
比较同一地区、同一天气类型在两个年份的臭氧AQI前后变化。
两个年份使用同一套窗口设置，窗口由程序管理。

地区、天气类型和污染物优先使用本轮明确内容，
没有明确内容时继承当前研究条件。

年份方向：
- “2024年与2023年相比”：目标年2024，基准年2023。
- “2023年相比2024年”：目标年2023，基准年2024。
- 只并列说“比较2023年和2024年”：
  默认较早年为基准，较晚年为目标，回答时说明方向。
- “和上一年比”：目标年使用当前研究年份，
  基准年为目标年的前一年。
- 年份无法确定时返回 null，不使用示例年份补齐。

此工具：
query_year=null
event_names=[]
baseline_year、target_year 分别填写两个年份。

只支持两个单独年份、一个地区、一个天气类型、臭氧AQI。
不支持月份比较、多地区比较、其他污染物或因果判断。
参数缺失时仍选择此工具，由程序检查缺失项。


5. read_saved_analysis
读取本会话已保存的分析记录，不重新计算。
回顾刚才的数值、询问完整记录数量、
解释为什么记录被排除，都选择此工具。

“刚才”“最近一次”：
saved_record_id=null。

明确指定旧年份、窗口或记录ID时：
从保存目录选择匹配的记录ID，不虚构ID。

没有保存结果时，仍选择此工具，由程序说明没有记录。

询问记录数量或筛选原因：
saved_result_view="coverage"。

其他结果回顾：
saved_result_view="summary"。

此工具：
event_names=[]
query_region=null
query_year=null
query_event_name=null
query_pollutant=null
baseline_year=null
target_year=null

明确要求重算、重新分析、换条件或窗口重新计算时，
选择原来的分析或比较工具。

例：刚才有多少条记录能完整比较？
工具：read_saved_analysis
saved_record_id=null
saved_result_view="coverage"

例：为什么539条最后只用了177条？
工具：read_saved_analysis
saved_record_id=null
saved_result_view="coverage"

例：刚才的AQI变化量是多少？
工具：read_saved_analysis
saved_record_id=null
saved_result_view="summary"

6. get_weather_forecast

待完成天气查询的接续规则：
当天气上下文pending_query不为空，用户本轮补充城市、州或选择候选时，
必须选择get_weather_forecast，即使本轮没有天气、下雨等词。

城市名称的不同中文写法也应识别：
例如休斯敦、休斯顿均对应Houston。
weather_city_original必须保留本轮实际原文，
不能替换成另一种中文写法。

本轮未提日期时，weather_day沿用pending_query.weather_day。
近期预报不需要历史研究年份和天气事件类型。

如果用户改问AQI概念、明确提出历史分析或其他独立问题，
按新问题选择工具，不得因为存在待完成天气查询就强行查询预报。

例：pending_query为
{"city_name":null,"state_name":"Texas","weather_day":"tomorrow"}

用户：休斯顿
工具：get_weather_forecast
weather_city_original="休斯顿"
weather_city="Houston"
weather_state="Texas"
weather_city_id=null
weather_day="tomorrow"
event_names=[]
query_region=null
query_year=null

同样存在上述pending_query时，用户问：AQI是什么意思？
工具：none


查询美国城市的当前可用NWS近期预报，不是历史天气查询。
适用于近期天气、降水概率、今天/明天/后天天气，
以及天气地点候选选择。

天气地点独立于历史AQI研究范围。
不能把研究年份当作预报年份。

weather_city_original：
本轮具体城市的原文片段，必须逐字来自问题。

weather_city：
对应英文检索名，不能把整个州当作城市。

weather_state：
英文州名；没有冲突时可用研究州作为定位提示。

没有明确新城市时：
weather_city_original和weather_city都为null。
程序会沿用已保存的天气地点；
没有已保存地点时会提示补充城市。
用户只给州名时，不能拿示例城市补齐。

weather_city_id：
只能从天气候选目录选择，不得虚构ID或经纬度。

weather_day：
近期或未指定available；
今天today；
明天tomorrow；
后天day_after_tomorrow。
选择城市候选时，延续pending_query中原请求的weather_day。

本工具的历史分析参数均为null，event_names=[]。
请求过去日期的日常天气，不选择这个预报工具。
不能用它预测AQI或声称覆盖接口范围以外的整月天气。

例：休斯敦，得州，最近会下雨吗？
工具：get_weather_forecast
weather_city_original="休斯敦"
weather_city="Houston"
weather_state="Texas"
weather_city_id=null
weather_day="available"

天气上下文已经选择Houston, Texas时：
问题：那明天呢？
工具：get_weather_forecast
weather_city_original=null
weather_city=null
weather_state=null
weather_city_id=null
weather_day="tomorrow"

问题：得州下周会下雨吗？
工具：get_weather_forecast
weather_city_original=null
weather_city=null
weather_state="Texas"
weather_city_id=null
weather_day="available"

问题：选第1项。
有待选天气候选时，选择get_weather_forecast，
weather_city_id填写目录中第一项的真实city_id。

7. none
一般概念解释、单纯修改条件、历史日常天气查询，
其他污染物分析，或超出上述工具能力的问题。

选择其他工具时，baseline_year、target_year 都返回 null。

event_names 规则：
- 查询全部类型时返回 []。
- 检查具体类型时，提取用户原文中的名称。
- 每个名称必须是问题中连续出现的原文。
- 不翻译、不增加前缀、冒号或解释。
- 多个名称拆成独立元素。
- 选择 none 时返回 []。

选择 list_weather_events 或 none 时，
query_region 和 query_year 都返回 null。

示例：

问题：内置数据有哪些天气事件类型？
工具：list_weather_events
event_names=[]
query_region=null
query_year=null

问题：内置数据有没有冰雹和野火？
工具：list_weather_events
event_names=["冰雹", "野火"]
query_region=null
query_year=null

问题：得州2023年有哪些天气事件类型？
工具：query_weather_events
event_names=[]
query_region="得州"
query_year=2023

当前研究地区为得州、年份为2023时：
问题：当地这一年有哪些天气事件类型？
工具：query_weather_events
event_names=[]
query_region="得州"
query_year=2023

问题：得州有哪些天气事件类型？
工具：query_weather_events
query_region="得州"
query_year 使用当前研究年份；没有则为 null。

问题：AQI是什么意思？
工具：none
event_names=[]
query_region=null
query_year=null

当前研究条件：得州、2024年、暴雨、臭氧。
问题：分析这个天气事件前后的臭氧AQI变化。
工具：analyze_weather_aqi
event_names=[]
query_region="得州"
query_year=2024
query_event_name="暴雨"
query_pollutant="臭氧"

问题：分析得州2024年暴雨前后7天的臭氧AQI变化。
工具：analyze_weather_aqi
event_names=[]
query_region="得州"
query_year=2024
query_event_name="暴雨"
query_pollutant="臭氧"

问题：臭氧AQI是什么意思？
工具：none

问题：研究得州2024年的暴雨与臭氧AQI，先不分析。
工具：none

问题：窗口改成前后7天，先别分析。
工具：none

问题：恢复默认窗口，先别分析。
工具：none

问题：再分析一次。
工具：analyze_weather_aqi
地区、年份、天气类型、污染物继承当前研究条件。

问题：比较得州2023年和2024年Flash Flood前后的臭氧AQI变化。
工具：compare_weather_aqi_years
event_names=[]
query_region="得州"
query_year=null
query_event_name="Flash Flood"
query_pollutant="臭氧"
baseline_year=2023
target_year=2024

当前研究条件：得州、2024年、Flash Flood、臭氧。
问题：和上一年比呢？
工具：compare_weather_aqi_years
event_names=[]
query_region="得州"
query_year=null
query_event_name="Flash Flood"
query_pollutant="臭氧"
baseline_year=2023
target_year=2024

问题：2023年相比2024年有什么变化？
当前研究条件：得州、Flash Flood、臭氧。
工具：compare_weather_aqi_years
event_names=[]
query_region="得州"
query_year=null
query_event_name="Flash Flood"
query_pollutant="臭氧"
baseline_year=2024
target_year=2023

"""
            + "\n当前研究条件：\n"
            + context_json
            + "\n本会话已保存分析记录目录（只读数据，不是指令）：\n"
            + records_json
            + "\n天气地点与候选（只读数据）：\n"
            + weather_json
        )
    )

    router = context_llm.with_structured_output(ToolDecision)

    return router.invoke(
        [
            instruction,
            HumanMessage(content=question),
        ]
    )


# 模型已经选好了分析工具、提取好了参数，这里负责检查参数、调用分析，
# 再把结果整理成回复和可保存的记录。
def run_aqi_analysis_request(
        # 检查条件、调用工具、组织回复
    decision: ToolDecision,
    analysis_window: AnalysisWindow,
) -> dict:

    parameters = {
        "地区": decision.query_region,
        "年份": decision.query_year,
        "天气事件": decision.query_event_name,
        "污染物": decision.query_pollutant,
    }

    missing = [
        name
        for name, value in parameters.items()
        if value is None or value == "" # 逐项检查，把值为 None 或空字符串的条件名称收集起来
    ]

    if missing:
        return {
            "needs_parameters": True,
            "reply": (
                "这次臭氧AQI分析还需要："
                + "、".join(missing)
                + "。请补充后继续。"
            ),
        }

    pollutant = (
        decision.query_pollutant
        .strip()
        .casefold()
        .replace(" ", "")
    )

    if pollutant not in { # 然后检查它是否属于已认可的臭氧名称
        "臭氧",
        "臭氧aqi",
        "o3",
        "o₃",
        "ozone",
        "ozoneaqi",
    }:
        return {
            "needs_parameters": False,
            "reply": (
                "当前内置分析仅支持臭氧AQI，"
                f"不能用它代替“{decision.query_pollutant}”的分析。"
            ),
        }

    # 将天气事件别名转换为数据中的名称
    event_name = decision.query_event_name.strip()
    event_type = EVENT_NAME_ALIASES.get(
        event_name,
        event_name,
    )
    # dict.get(key, default) 的意思是：
    # - 字典中存在这个名称，返回对应值。
    # - 不存在，就保留原名称。

    # 读取当前窗口，并打印执行参数
    before_days = analysis_window.before_days
    after_days = analysis_window.after_days

    print(
        f"[分析参数] 地区={decision.query_region}，"
        f"年份={decision.query_year}，"
        f"事件={event_type}，"
        f"前{before_days}天、后{after_days}天。",
        flush=True,
    )

    try:
        result = analyze_weather_aqi.invoke( # 	Agent 工具接口，整理分析状态和说明
            {
                "region": decision.query_region,
                "year": decision.query_year,
                "event_type": event_type,
                "before_days": before_days,
                "after_days": after_days,
            }
        )

    except (OSError, ValueError, KeyError) as error: # 把预期错误转换为可读回复
        return {
            "needs_parameters": False,
            "reply": f"臭氧AQI分析未完成：{error}。",
        }

    lines = [ # 将分析结果整理成文字
        f"分析地区：{result['region']}。",
        f"事件开始年份：{result['year']}。",
        f"天气类型：{event_name} → {result['event_type']}。",
        "分析指标：臭氧AQI。",
        (
            f"窗口：事件开始前{result['before_days']}天、"
            f"结束后{result['after_days']}天，"
            "不包含事件起止当天。"
        ),
        (
            f"匹配事件记录：{result['matched_event_rows']}行；"
            f"其中县级记录：{result['county_event_rows']}行，"
            f"排除非县级记录：{result['excluded_non_county_rows']}行。"
        ),
        (
            f"无同县站点记录：{result['rows_without_monitor']}行；"
            "有站点但没有完整窗口比较的记录："
            f"{result['rows_without_complete_window']}行。"
        ),
        (
            "可完整比较的事件记录："
            f"{result['rows_with_complete_comparison']}行。"
        ),
    ]

    if result["status"] == "ok": # 只有分析状态正常时，才显示变化量
        change = result["mean_event_row_aqi_change"]

        lines.append(
            f"平均AQI变化量（后减前）：{change:+.2f}点。"
        )

    lines.append(result["message"])

    return {
        "needs_parameters": False,
        "reply": "\n".join(lines),
        "confirmed_pollutant": "臭氧",
        "result_record": make_analysis_record(
            "analyze_weather_aqi",
            result,
        ),
    }


# 确定跨年比较中的两个年份，以及谁是基准年、谁是目标年。
def resolve_comparison_years(
    question: str,
    current_year: int | None,
) -> tuple[int, int]:

    years = { # 从问题中提取不同的四位年份
        int(value)
        for value in re.findall(
            r"(?<!\d)(\d{4})(?!\d)",
            question,
        )
    }

    if len(years) == 2: # 恰好两个年份时，先默认早年作基准
        baseline_year, target_year = sorted(years)

        directions = [
            r"(\d{4})\s*年?\s*(?:与|和)\s*(\d{4})\s*年?\s*相比",
            r"(\d{4})\s*年?\s*(?:相比于|相比|相较于|比)\s*(\d{4})",
        ]

        for pattern in directions:
            match = re.search(pattern, question)

            if match:
                target_year = int(match.group(1))
                baseline_year = int(match.group(2))
                break

        reference = re.search(
            r"(\d{4})\s*年?\s*(?:为|作为)\s*(?:基准|参照)",
            question,
        )

        if reference:
            baseline_year = int(reference.group(1))
            target_year = next( # 取出“不等于基准年”的那个年份。
                year for year in years
                if year != baseline_year
            )

        return baseline_year, target_year

    if len(years) <= 1 and any(
            word in question for word in ("上一年", "去年", "前面一年", "前一年")
    ):

        target_year = (
            next(iter(years))
            if years
            else current_year
        )

        if target_year is not None and target_year > 1:
            return target_year - 1, target_year

    raise ValueError(
        "请指定两个不同年份；"
        "或先设定研究年份，再说“和上一年比”。"
    )


def run_year_comparison_request(
    decision: ToolDecision,
    question: str,
    research_context: dict,
    analysis_window: AnalysisWindow,
) -> dict:

    try:
        baseline_year, target_year = resolve_comparison_years(
            question,
            research_context.get("year"),
        )
    except ValueError as error:
        return {
            "needs_parameters": True,
            "reply": str(error),
        }

    parameters = {
        "地区": decision.query_region,
        "天气事件": decision.query_event_name,
        "污染物": decision.query_pollutant,
    }

    missing = [
        name
        for name, value in parameters.items()
        if not value
    ]

    if missing:
        return {
            "needs_parameters": True,
            "reply": (
                "跨年比较还需要："
                + "、".join(missing)
                + "。"
            ),
        }

    pollutant = (
        decision.query_pollutant
        .strip()
        .casefold()
        .replace(" ", "")
    )

    if pollutant not in {
        "臭氧", "臭氧aqi", "o3", "o₃", "ozone", "ozoneaqi"
    }:
        return {
            "needs_parameters": False,
            "reply": "当前跨年比较仅支持臭氧AQI。",
        }

    event_name = decision.query_event_name.strip()
    event_type = EVENT_NAME_ALIASES.get(
        event_name,
        event_name,
    )

    print(
        f"[跨年参数] {decision.query_region}，{event_type}，"
        f"基准年{baseline_year}，目标年{target_year}，"
        f"前{analysis_window.before_days}天、"
        f"后{analysis_window.after_days}天。",
        flush=True,
    )

    try:
        result = compare_weather_aqi_years.invoke(
            {
                "region": decision.query_region,
                "baseline_year": baseline_year,
                "target_year": target_year,
                "event_type": event_type,
                "before_days": analysis_window.before_days,
                "after_days": analysis_window.after_days,
            }
        )
    except (OSError, ValueError, KeyError) as error:
        return {
            "needs_parameters": False,
            "reply": f"跨年比较未完成：{error}。",
        }

    lines = [
        f"比较范围：{result['region']}，"
        f"{result['event_type']}，臭氧AQI。",
        f"基准年：{baseline_year}；目标年：{target_year}。",
        f"统一窗口：开始前{result['before_days']}天、"
        f"结束后{result['after_days']}天，"
        "不包含事件起止当天。",
    ]

    for label, title in (
        ("baseline", "基准年"),
        ("target", "目标年"),
    ):
        annual = result[label]
        change = annual["mean_event_row_aqi_change"]

        change_text = (
            "无法计算"
            if change is None
            else f"{change:+.4f}点"
        )

        lines.append(
            f"{title}{annual['year']}："
            f"匹配{annual['matched_event_rows']}条，"
            f"完整比较{annual['rows_with_complete_comparison']}条，"
            f"平均前后变化量：{change_text}。"
        )

    difference = result["change_difference"]

    if difference is None:
        lines.append("数据不足，无法计算年份间差值。")
    else:
        lines.append(
            "年份间差值（目标年减基准年）："
            f"{difference:+.4f}点。"
        )

    lines.append(result["message"])

    return {
        "needs_parameters": False,
        "reply": "\n".join(lines),
        "result_record": make_analysis_record(
            "compare_weather_aqi_years",
            result,
        ),
    }


    # 以user说的：当前研究2024年，用户说：“把年份改成2023年，顺便解释一下AQI是什么。” 为例子
def chat_with_user(state: AnalysisState) -> dict:

    new_analysis_results = []

    weather_context = dict(state.get("weather_context") or {})

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

    analysis_window = AnalysisWindow.model_validate(
        state.get("analysis_window", {})
    )

    pending_weather_query = state.get("pending_weather_query")
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
    跨年比较中提到两个年份，不等于要求修改当前研究年份。
    例如“比较2023年和2024年”，不要生成 year 的 set 操作。
    只有用户明确要求修改研究条件时，才修改对应条件。
    比较请求应完整保留在 answer_request 中。

    不要把“先不分析”当作待执行的分析请求。
    没有明确修改要求时，changes=[]。
    假设问题中明确说先别修改，也不修改。

    明确要求分析、计算或查询时，answer_request 保留完整请求。
    即使同时提取了研究条件，也不能省略执行请求。
    “分析……”与“研究……先不分析”必须区分。
    用户明确提到臭氧时，应提取 pollutant="臭氧"。

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

    回顾已保存分析、询问原结果中的数量或筛选原因时，
    不因提到旧年份、旧事件或旧窗口就生成条件或窗口修改。
    只有明确要求修改条件时才提取变更。

    近期天气查询的城市和时间不修改历史研究条件。
    天气问题中的“未来7天”等天数，不属于事件前后分析窗口。
    只有用户明确要求修改研究条件或分析窗口时，才提取对应操作。

    近期天气地点补充、候选选择由天气工具处理，
    pending_query_action="keep"，answer_request保留本轮原话。
    不要恢复或取消历史AQI待完成查询，也不要因此修改历史研究条件。

    窗口操作规则：
    “窗口改成前后7天，先别分析”：
    changes=[]，window_action="update"，answer_request=null。

    “恢复默认窗口，先别分析”：
    changes=[]，window_action="reset"，answer_request=null。

    “再分析一次”：
    window_action="keep"，answer_request保留原话。

    “前后7天的窗口合理吗”：
    window_action="keep"，answer_request保留问题。

    单纯修改窗口不修改地区、年份、事件或污染物。

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
                + "\n独立的近期天气状态："
                + json.dumps(
            {
                "location": weather_context.get("location"),
                "candidates": weather_context.get("candidates", []),
                "pending_query": weather_context.get("pending_query"),
            },
            ensure_ascii=False,
        )
                + "\n待完成的天气查询："
                + str(pending_weather_query)
                + """
    处理待完成查询：
    没有待完成查询时，pending_query_action="keep"。

    有待完成查询时：
    - “那就查2023年的”：提取年份修改，
      pending_query_action="resume"，answer_request=null。
    - “继续刚才的查询”：pending_query_action="resume"，
      answer_request=null。
    - “年份改成2023年，先别查”：只修改年份，
      pending_query_action="keep"，answer_request=null。
    - “AQI是什么意思”：正常保留概念问题，
      pending_query_action="keep"。
    - “不查了，取消刚才的查询”：pending_query_action="cancel"，
      answer_request=null。

    不要仅因为存在待完成查询就自动执行它。
    
    用户未指定年份时，不得补充年份，也不得使用示例中的年份。

    例：研究得州的暴雨与臭氧AQI，先不分析。
    {"changes":[{"field":"region","operation":"set","value":"得州"},
    {"field":"event_description","operation":"set","value":"暴雨"},
    {"field":"pollutant","operation":"set","value":"臭氧"}],
    "answer_request":null,"pending_query_action":"keep"}
    """
        )
    )

    context_model = context_llm.with_structured_output( # 调用4b模型，理解本轮要求
        ResearchAction
    )
    if is_weather_cancellation(
            str(state["messages"][-1].content),
            weather_context,
    ):
        action = ResearchAction(
            changes=[],
            answer_request=str(state["messages"][-1].content),
        )
    else:
        action = context_model.invoke(
            [context_instruction, state["messages"][-1]]
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

    latest_user_text = str(state["messages"][-1].content)

    mentioned_years = {
        int(value)
        for value in re.findall(
            r"(?<!\d)(\d{4})(?!\d)",
            latest_user_text,
        )
    }

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

                if len(mentioned_years) > 1:
                    explicit_update = re.search(
                        r"(?:研究)?年份\s*"
                        r"(?:改成|改为|设为|设置为)\s*(\d{4})",
                        latest_user_text,
                    )

                    if (
                        explicit_update is None
                        or change.value != int(explicit_update.group(1))
                    ):
                        continue

                if valid_value and change.value not in mentioned_years:
                    update_notes.append(
                        "本轮没有明确提供该数字年份，"
                        "未执行模型提出的年份修改，保留原条件。"
                    )
                    continue

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
            "项目已有内置历史天气与臭氧AQI数据。"
            "当前支持全局事件类型查询、地区年份事件类型查询，"
            "以及指定天气事件前后的臭氧AQI比较。"
            "实际查询与计算由程序的工具分支执行，"
            "不能在普通问答中声称已执行工具或编造计算结果。"
            "已接入美国城市的NWS近期天气预报，"
            "实际范围以接口返回时间为准。"
            "尚不支持历史逐日天气、其他污染物计算和因果判断。"
            "可以解释一般概念，但不能编造数据、计算结果或最新信息。"
            "不必每轮重复介绍能力限制。"
            "用户询问概念时，直接解释概念，"
            "不要强行联系当前研究，也不要添加未经验证的天气与污染关系。"
            "用户说先不分析时，简短确认保存的条件即可。"
            "项目已有内置历史数据，不要默认要求用户上传或提供数据。"
            "用户要求实际计算时必须由工具执行，"
            "工具未返回结果时不能编造计算结论。"
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
    routing_question = latest_user_text

    if action.pending_query_action == "cancel":
        pending_weather_query = None
        reply_parts.append("待完成的天气查询已取消。")

    elif action.pending_query_action == "resume":
        if pending_weather_query is not None:
            routing_question = pending_weather_query

            if not answer_request:
                answer_request = pending_weather_query
        else:
            reply_parts.append("当前没有待完成的天气查询。")

    decision = None

    if is_weather_cancellation(
            latest_user_text,
            weather_context,
    ):
        decision = ToolDecision(
            reason="取消待完成的近期天气查询",
            tool_name="get_weather_forecast",
            event_names=[],
            query_region=None,
            query_year=None,
        )

    elif not (
            action.pending_query_action == "cancel"
            and not answer_request
    ):
        decision = choose_tool(
            routing_question,
            current_snapshot,
            analysis_results=state.get("analysis_results", []),
            weather_context=weather_context,
        )

    if decision is not None and decision.tool_name != "none":
        answer_request = routing_question

    window_command = re.fullmatch(
        r"(?:把)?(?:分析)?窗口\s*(?:改成|改为|设为|设置为)\s*"
        r"(?:前后\s*\d+\s*天|"
        r"前\s*\d+\s*天(?:[、，,\s]*后\s*\d+\s*天)?|"
        r"后\s*\d+\s*天)"
        r"(?:[，,]\s*(?:先别分析|先不分析|不分析))?[。.!！]?",
        latest_user_text.strip(),
    )

    if window_command is not None:
        action.window_action = "update"
        answer_request = None

        if decision is not None:
            decision.tool_name = "none"

    window_error = False

    try:
        proposed_window = analysis_window

        if action.window_action == "reset":
            proposed_window = AnalysisWindow()

        elif action.window_action == "update" or (
            decision is not None
            and decision.tool_name in {
                "analyze_weather_aqi",
                "compare_weather_aqi_years",
            }
            and re.search(
                r"(?:前后|前|后)\s*\d+\s*天",
                latest_user_text,
            )
        ):
            proposed_window = update_analysis_window(
                latest_user_text,
                analysis_window,
            )

        if proposed_window != analysis_window:
            analysis_window = proposed_window

            reply_parts.append(
                f"分析窗口已设为：开始前{analysis_window.before_days}天、"
                f"结束后{analysis_window.after_days}天。"
            )

        elif action.window_action in {"update", "reset"}:
            reply_parts.append("分析窗口与当前设置一致。")

    except ValueError as error:
        window_error = True
        reply_parts.append(str(error))

    if (
        window_error
        and decision is not None
        and decision.tool_name == "analyze_weather_aqi"
    ):
        answer_request = None

    if answer_request is not None and answer_request.strip() != "":

        if decision.tool_name in {
            "query_weather_events",
            "analyze_weather_aqi",
        }:
            explicit_years = {
                int(value)
                for value in re.findall(
                    r"(?<!\d)(\d{4})(?!\d)",
                    answer_request,
                )
            }

            if len(explicit_years) == 1:
                decision.query_year = next(iter(explicit_years))

            elif not explicit_years:
                decision.query_year = current_snapshot["year"]

            else:
                decision.query_year = None
                reply_parts.append(
                    "本轮包含多个年份，当前工具只支持单个年份查询。"
                )
        if decision.tool_name in {
            "analyze_weather_aqi",
            "compare_weather_aqi_years",
        }:
            inherited_fields = {
                "query_region": "region",
                "query_event_name": "event_description",
                "query_pollutant": "pollutant",
            }

            for query_field, context_field in inherited_fields.items():
                candidate_value = getattr(decision, query_field)

                explicitly_mentioned = (
                    isinstance(candidate_value, str)
                    and candidate_value.strip() != ""
                    and candidate_value.strip().casefold()
                    in routing_question.casefold()
                )

                if not explicitly_mentioned:
                    setattr(
                        decision,
                        query_field,
                        current_snapshot[context_field],
                    )

        if decision.tool_name == "get_weather_forecast":
            weather_response = run_weather_request(
                question=latest_user_text,
                parameters=decision.model_dump(),
                context=weather_context,
                forecast_tool=get_weather_forecast,
            )

            weather_context = weather_response["weather_context"]
            reply_parts.append(weather_response["reply"])

        elif decision.tool_name == "read_saved_analysis":
            saved_reply = read_saved_analysis.invoke(
                {
                    "records": state.get("analysis_results", []),
                    "record_id": decision.saved_record_id,
                    "view": decision.saved_result_view,
                }
            )

            reply_parts.append(saved_reply)

        elif decision.tool_name == "list_weather_events":
            print("\n[工具] 正在读取内置天气事件类型……")

            try:
                event_types = list_weather_events.invoke({})

                scope_text = (
                    "查询范围：整个内置天气事件数据集，"
                    "未按地区或年份筛选。"
                )

                if decision.event_names:
                    type_lookup = {
                        name.strip().casefold(): name  # 把文件里的真实类型整理成方便查询的字典
                        for name in event_types
                    }

                    conclusions = []

                    for requested_name in decision.event_names:
                        cleaned_name = requested_name.strip()

                        if cleaned_name.casefold() not in answer_request.casefold():
                            reply_parts.append(
                                "事件名称提取出现错误："
                                f"“{cleaned_name}”不在本轮问题原文中。"
                                "该名称未执行匹配，不能据此判断数据中是否存在。"
                            )
                            continue

                        if cleaned_name.casefold() not in answer_request.casefold():
                            reply_parts.append(
                                "事件名称提取出现错误："
                                f"“{cleaned_name}”不在本轮问题原文中。"
                                "该名称未执行匹配，不能据此判断数据中是否存在。"
                            )
                            continue

                        lookup_name = EVENT_NAME_ALIASES.get(
                        # 查对应表：输入“龙卷风”就得到 "Tornado"；输入本来就是 "Tornado"，就保留原名。
                            cleaned_name,
                            cleaned_name,
                        )
                        # 用户问“有没有龙卷风”
                        # → 模型提取“龙卷风”
                        # → 对应表转换为 Tornado
                        # → 在真实类型列表中匹配
                        # → 输出“内置数据包含事件类型：Tornado”
                        # 判断名称是否存在，并生成结论。
                        matched_name = type_lookup.get(
                        # 再去文件实际读出的类型列表里核对。casefold() 用于忽略英文大小写，
                        # 因此 tornado 也能匹配 Tornado。
                            lookup_name.casefold()
                        )

                        if matched_name is not None:
                            conclusions.append(
                                f"查询名称：{cleaned_name}。\n"
                                f"内置数据包含事件类型：{matched_name}。"
                            )
                        else:
                            conclusions.append(
                                f"未找到与“{requested_name}”"
                                "名称完全匹配的事件类型。"
                                "这里只做名称匹配，"
                                "尚未判断中文名称或同义词是否对应其他类型。"
                            )

                    reply_parts.append(
                        "\n".join(conclusions) + "\n\n" + scope_text
                    )

                else: # 用户问全部类型时，由 Python 生成列表和数量。
                    event_list = "\n".join(
                        f"{index}. {name}"
                        for index, name in enumerate(  # enumerate() 给类型编号
                            event_types, start=1
                        )
                    )

                    reply_parts.append(
                        f"{scope_text}\n"
                        f"共 {len(event_types)} 种，"
                        "以下保留数据中的原始名称：\n\n"
                        f"{event_list}"
                    )

            except (OSError, ValueError, KeyError) as error:
                reply_parts.append(
                    f"读取内置天气事件类型失败：{error}。"
                    "本轮没有获得有效查询结果。"
                )

        elif decision.tool_name == "query_weather_events":
            missing_fields = []

            if not decision.query_region:
                missing_fields.append("地区")

            if decision.query_year is None:
                missing_fields.append("年份")

            if missing_fields:
                pending_weather_query = answer_request

                reply_parts.append(
                    "这次查询还需要："
                    + "、".join(missing_fields)
                    + "。请补充后再查询。"
                )

            else:
                print("\n[工具] 正在按地区和年份筛选天气事件……")

                try:
                    result = query_weather_events.invoke(
                        {
                            "region": decision.query_region,
                            "year": decision.query_year,
                        }
                    )
                    pending_weather_query = None

                    scope_text = (
                        f"查询地区：{result['region']}；"
                        f"州代码：{result['state_fips']}。\n"
                        f"查询年份：{result['year']}；"
                        f"按{result['year_basis']}筛选。\n"
                        f"匹配记录：{result['matched_row_count']}行；"
                        f"事件类型：{result['event_type_count']}种。"
                    )

                    reply_parts.append(scope_text)

                    if result["matched_row_count"] == 0:
                        reply_parts.append(
                            "内置数据中没有匹配该地区和年份的记录。"
                            "这不代表现实中没有发生天气事件。"
                        )

                    elif decision.event_names:
                        type_lookup = {
                            name.casefold(): name
                            for name in result["event_types"]
                        }

                        for requested_name in decision.event_names:
                            cleaned_name = requested_name.strip()

                            if (
                                not cleaned_name
                                or cleaned_name.casefold()
                                not in answer_request.casefold()
                            ):
                                reply_parts.append(
                                    f"事件名称“{requested_name}”"
                                    "提取异常，未执行该名称的匹配。"
                                )
                                continue

                            lookup_name = EVENT_NAME_ALIASES.get(
                                cleaned_name,
                                cleaned_name,
                            )

                            matched_name = type_lookup.get(
                                lookup_name.casefold()
                            )

                            if matched_name is not None:
                                reply_parts.append(
                                    f"该查询范围内包含事件类型："
                                    f"{matched_name}"
                                    f"（查询名称：{cleaned_name}）。"
                                )
                            else:
                                reply_parts.append(
                                    f"该查询范围内未找到与"
                                    f"“{cleaned_name}”对应的类型名称。"
                                    "当前仅支持原名和已配置的中文别名匹配。"
                                )

                    else:
                        event_list = "\n".join(
                            f"{index}. {name}"
                            for index, name in enumerate(
                                result["event_types"],
                                start=1,
                            )
                        )

                        reply_parts.append(
                            "以下保留数据中的原始名称：\n"
                            + event_list
                        )

                except (OSError, ValueError, KeyError) as error:
                    reply_parts.append(
                        f"按地区和年份查询失败：{error}。"
                        "本轮未获得有效查询结果。"
                    )

        elif decision.tool_name == "compare_weather_aqi_years":
            comparison_response = run_year_comparison_request(
                decision=decision,
                question=answer_request,
                research_context=current_snapshot,
                analysis_window=analysis_window,
            )

            reply_parts.append(comparison_response["reply"])

            if comparison_response.get("result_record") is not None:
                new_analysis_results.append(
                    comparison_response["result_record"]
                )

            pending_weather_query = (
                answer_request
                if comparison_response["needs_parameters"]
                else None
            )


        elif decision.tool_name == "analyze_weather_aqi":
            analysis_response = run_aqi_analysis_request(
                decision=decision,
                analysis_window=analysis_window,
            )

            reply_parts.append(analysis_response["reply"])

            if analysis_response.get("result_record") is not None:
                new_analysis_results.append(
                    analysis_response["result_record"]
                )

            explicitly_cleared_pollutant = any(
                change.field == "pollutant"
                and change.operation == "clear"
                for change in action.changes
            )

            if (
                analysis_response.get("confirmed_pollutant") == "臭氧"
                and current_snapshot["pollutant"] is None
                and changed_fields
                and not explicitly_cleared_pollutant
                and re.search(
                    r"臭氧|(?<![A-Za-z0-9])(?:o3|o₃|ozone)(?![A-Za-z0-9])",
                    latest_user_text,
                    flags=re.IGNORECASE,
                )
            ):
                current_snapshot["pollutant"] = "臭氧"
                reply_parts.append("污染物已设为：臭氧。")

            if analysis_response["needs_parameters"]:
                pending_weather_query = answer_request
            else:
                pending_weather_query = None

        else:
            question_instruction = SystemMessage(
                content=(
                        answer_instruction.content
                        + "\n条件操作的确认由程序单独提供。"
                          "本次只回答用户的问题，不重复确认条件。"
                          "本轮没有执行数据查询工具。"
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

    if current_snapshot != previous_snapshot:
        # 研究条件确实改变了，才记录一条新的历史快照。
        if not research_history and any(
            value is not None
            for value in previous_snapshot.values()
        ):
            research_history.append(previous_snapshot.copy())

        research_history.append(current_snapshot.copy())

    # 合成回答，返回本轮状态更新
    response = AIMessage(
        content="\n\n".join(reply_parts)
    )

    return {
        "messages": [response],
        "research_context": current_snapshot,
        "research_history": research_history,
        "pending_weather_query": pending_weather_query,
        "analysis_window": analysis_window.model_dump(),
        "analysis_results": new_analysis_results,
        "weather_context": weather_context,
    }
# |     返回内容        |            后续作用 |
# |--------------------|--------------------------------------|
# | `messages`         | 新回答通过 `add_messages` 合并进聊天记录 |
# | `research_context` | 保存本轮最新条件 |
# | `research_history` | 保存更新后的历史快照列表 |


analysis_graph = build_uploaded_file_analysis_graph(AnalysisState, llm)


chat_builder = StateGraph(AnalysisState)
chat_builder.add_node("chat_with_user", chat_with_user)

chat_builder.add_edge(START, "chat_with_user")
chat_builder.add_edge("chat_with_user", END)

graph = chat_builder.compile(
    checkpointer=InMemorySaver(),
)


# 第一次invoke
if __name__ == "__main__":
    with ExitStack() as storage_stack:
        sqlite_checkpointer = None
        session_name = ""
        is_temporary = True
        persistent_graph = chat_builder.compile(checkpointer=InMemorySaver())
        config = {"configurable": {"thread_id": f"temporary:{uuid4()}"}}

        print("天气与空气质量助手已启动，当前为临时聊天。")
        print("直接输入问题；临时聊天退出后不保存。")
        print("命令：/恢复 会话名；/会话 会话名（创建或恢复）；/新建；退出。")

        while True:
            user_input = input("\n你：").strip()

            if user_input.lower() in {"退出", "exit", "quit", "0"}:
                if is_temporary:
                    print("临时聊天已结束，本次临时聊天未保存。")
                else:
                    print("本次对话结束，会话已保存在本地。")
                break

            if user_input == "":
                continue

            command_parts = user_input.split(maxsplit=1)
            command = command_parts[0]

            if command == "/新建":
                if len(command_parts) != 1:
                    print("用法：/新建")
                    continue
                session_name = ""
                is_temporary = True
                persistent_graph = chat_builder.compile(checkpointer=InMemorySaver())
                config = {"configurable": {"thread_id": f"temporary:{uuid4()}"}}
                print("已开始全新临时聊天；已有命名会话仍保存在本地。")
                continue

            if command in {"/恢复", "/会话"}:
                if len(command_parts) != 2:
                    print(f"用法：{command} 会话名，例如：{command} texas-study")
                    continue

                requested_name = command_parts[1].strip()
                requested_config = {
                    "configurable": {"thread_id": f"aqi:{requested_name}"}
                }
                if sqlite_checkpointer is None:
                    sqlite_checkpointer = storage_stack.enter_context(
                        open_session_checkpointer()
                    )
                requested_graph = chat_builder.compile(checkpointer=sqlite_checkpointer)
                saved_state = requested_graph.get_state(requested_config).values

                if not saved_state and command == "/恢复":
                    print("没有找到这个会话，当前聊天保持不变。")
                    print("如需创建，输入：/会话 会话名")
                    continue

                persistent_graph = requested_graph
                config = requested_config
                session_name = requested_name
                is_temporary = False

                if saved_state:
                    print(f"\n已恢复会话：{session_name}")
                    print(
                        f"已有聊天消息："
                        f"{len(saved_state.get('messages', []))}条"
                    )

                    print("\n已保存的研究条件：")
                    pprint(saved_state.get("research_context", {}))

                    print("\n已保存的分析窗口：")
                    pprint(saved_state.get("analysis_window", {}))

                    print("\n已选天气地点：")
                    pprint(
                        saved_state.get("weather_context", {}).get("location")
                    )

                    records = saved_state.get("analysis_results", [])
                    print(f"\n已保存分析结果：{len(records)}条")

                    if records:
                        print("\n最近一次分析参数：")
                        pprint(records[-1]["parameters"])

                        print("\n最近一次分析结果：")
                        pprint(records[-1]["result"])

                    pending = saved_state.get("pending_weather_query")

                    if pending:
                        print(f"\n待完成查询：{pending}")
                else:
                    print(f"\n已创建可保存会话：{session_name}")
                print("已切换会话；刚才聊天的内容不会合并到此会话。")
                continue

            if command.startswith("/"):
                print("支持的命令：/恢复 会话名；/会话 会话名；/新建；退出。")
                continue

            result = persistent_graph.invoke(
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

            print("\n当前分析窗口：")
            pprint(result.get("analysis_window", {}))

            print(
                "\n已保存分析结果："
                f"{len(result.get('analysis_results', []))}条"
            )

            print("\n已选天气地点：")
            pprint(
                result.get("weather_context", {}).get("location")
            )
