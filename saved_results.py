from typing import Literal

from langchain_core.tools import tool

# 整个文件就是让 Agent 知道之前做过哪些分析，再按需读取其中一份结果，不重新读 Excel、计算 AQI。

def saved_result_headers(records: list[dict]) -> list[dict]: # 提取记录目录
    return [
        {
            "record_id": record["record_id"],
            "created_at": record["created_at"],
            "tool_name": record["tool_name"],
            "parameters": record["parameters"],
        }
        for record in records
    ]


@tool
def read_saved_analysis( # 读取指定记录
    records: list[dict],
    record_id: str | None = None,
    view: Literal["summary", "coverage"] = "summary",
) -> str:
    """读取本会话保存的分析汇总，不读取Excel或重新计算。

    record_id为空时读取最近生成的记录。
    coverage解释筛选数量，summary回顾数值结果。
    """
    if not records:
        return "本会话尚无已保存的分析结果。"

    record = records[-1] if record_id is None else next(
        (
            item
            for item in records
            if item["record_id"] == record_id
        ),
        None,
    )

    if record is None:
        return "本会话中没有找到指定的分析记录，请核对记录ID。"

    # 取出分析条件和结果，准备文字列表
    parameters = record["parameters"] # 当时用什么条件分析
    result = record["result"]

    lines = [
        f"读取已保存记录：{record['record_id']}。",
        f"记录时间（UTC）：{record['created_at']}。",
        f"记录范围：{parameters['region']}，"
        f"{parameters['event_type']}，臭氧AQI。",
        f"记录窗口：开始前{parameters['before_days']}天、"
        f"结束后{parameters['after_days']}天。",
    ]

    is_comparison = "baseline" in result and "target" in result # 判断是单年分析，还是跨年比较

    annual_results = (
        [
            ("基准年", result["baseline"]),
            ("目标年", result["target"]),
        ]
        if is_comparison
        else [("分析年份", result)]
    )

    for title, annual in annual_results:
        total = annual["matched_event_rows"]
        complete = annual["rows_with_complete_comparison"]
        change = annual["mean_event_row_aqi_change"]

        lines.append(
            f"{title}{annual['year']}："
            f"匹配{total}条，完整比较{complete}条。"
        )

        if view == "coverage": # `coverage` 模式额外解释筛选过程，它把不能参与比较的事件分成三类
            excluded = annual["excluded_non_county_rows"] #非县级事件，不能直接按县匹配。
            no_monitor = annual["rows_without_monitor"] # 县级事件，但没有同县 AQI 站点。
            incomplete = annual["rows_without_complete_window"] # 有同县站点，但没有任何站点满足完整前后窗口。

            if total != excluded + no_monitor + incomplete + complete: # 然后验证数量是否对齐
                lines.append(
                    "记录中的筛选数量未能对齐，"
                    "不能据此完整解释筛选过程。"
                )
            else:
                lines.extend(
                    [
                        f"排除非县级记录{excluded}条；",
                        f"内置AQI数据无同县站点记录{no_monitor}条；",
                        f"有同县站点但无完整窗口比较{incomplete}条；",
                        f"因此：{total} - {excluded} - "
                        f"{no_monitor} - {incomplete} = {complete}条。",
                    ]
                )

        change_text = ( # 格式化变化量
            "无法计算"
            if change is None
            else f"{change:+.4f}点"
        )

        lines.append(f"平均事件前后变化量：{change_text}。")

    if is_comparison: # 跨年记录再显示年份间差值
        difference = result["change_difference"] # 目标年平均前后变化量 − 基准年平均前后变化量

        difference_text = (
            "无法计算"
            if difference is None
            else f"{difference:+.4f}点"
        )

        lines.append(
            f"年份间差值（目标年减基准年）：{difference_text}。"
        )

    lines.append(result.get("message", ""))
    lines.append("以上使用已保存的结果，本轮未重新计算。")

    return "\n".join(lines)