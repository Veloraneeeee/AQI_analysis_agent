from functools import lru_cache     # lru_cache 来实现内存缓存（In-Memory Caching）
                                    # lru_cache 缓存的是函数调用结果，而不是对话次数。
from pathlib import Path
from time import perf_counter

import pandas as pd


@lru_cache(maxsize=4)
def _read_excel_cached(
    source_path: str, # 不同文件或不同列组合，分别缓存
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


    # 确定文件位置，获取文件状态
    source = Path(file_path).resolve() # Path(file_path) 把路径转换成路径对象，`.resolve()` 将它转换成绝对路径。
    source_stat = source.stat() # source.stat() 获取文件的状态信息。

    started = perf_counter() # 适合计算一段操作花了多久
    previous_hits = _read_excel_cached.cache_info().hits # 缓存累计命中的次数

    table = _read_excel_cached( # 获取表格：命中缓存就复用，否则读取文件
        str(source),
        tuple(columns),
        source_stat.st_mtime_ns, # 文件最后修改时间，单位为纳秒
        source_stat.st_size, # 文件大小，单位为字节。文件修改时间或大小变化时，缓存键通常随之变化，触发重新读取。
    )

    # 判断本次是否用了缓存
    used_cache = (
        _read_excel_cached.cache_info().hits > previous_hits
    )
    # 调用前 3 次 → 调用后 4 次：本次命中了缓存，True
    # 调用前 3 次 → 调用后 3 次：本次重新读取，False

    result = table.copy()  # 返回副本，避免后续分析直接修改缓存里的原始 DataFrame。

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