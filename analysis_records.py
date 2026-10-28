from copy import deepcopy
from datetime import datetime, timezone
from uuid import uuid4

from data_sources import EVENT_FILE_PATH, AQI_FILE_PATH


def make_analysis_record(tool_name: str, result: dict) -> dict:
    parameter_names = [
        "region",
        "year",
        "baseline_year",
        "target_year",
        "event_type",
        "before_days",
        "after_days",
    ]

    parameters = {
        name: result[name]
        for name in parameter_names
        if name in result
    }
    parameters["pollutant"] = "臭氧"

    sources = []

    for path in (EVENT_FILE_PATH, AQI_FILE_PATH):
        file_info = path.stat()

        sources.append(
            {
                "file_name": path.name,
                "size_bytes": file_info.st_size,
                "modified_ns": file_info.st_mtime_ns,
            }
        )

    return {
        "record_id": str(uuid4()),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "tool_name": tool_name,
        "method_version": "event-window-v1",
        "parameters": parameters,
        "data_sources": sources,
        "result": deepcopy(result),
    }