from functools import lru_cache
from pathlib import Path
from time import perf_counter

import pandas as pd


@lru_cache(maxsize=4)
def _read_excel_cached(
    source_path: str,
    columns: tuple[str, ...],
    modified_ns: int,
    file_size: int,
) -> pd.DataFrame:
    print(
        f"[数据] 正在读取 {Path(source_path).name}……",
        flush=True,
    )

    return pd.read_excel(
        source_path,
        usecols=list(columns),
    )


def read_cached_excel(
    file_path: str | Path,
    columns: list[str],
) -> pd.DataFrame:
    source = Path(file_path).resolve()
    source_stat = source.stat()

    started = perf_counter()
    previous_hits = _read_excel_cached.cache_info().hits

    table = _read_excel_cached(
        str(source),
        tuple(columns),
        source_stat.st_mtime_ns,
        source_stat.st_size,
    )

    used_cache = (
        _read_excel_cached.cache_info().hits > previous_hits
    )

    result = table.copy()

    origin = (
        "复用内存缓存"
        if used_cache
        else "读取Excel完成"
    )

    print(
        f"[数据] {origin}：{source.name}，"
        f"{len(result)}行，"
        f"耗时{perf_counter() - started:.2f}秒。",
        flush=True,
    )

    return result