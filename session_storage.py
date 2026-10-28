from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from langgraph.checkpoint.sqlite import SqliteSaver


RUNTIME_DIR = Path(__file__).resolve().parent / "runtime"
CHECKPOINT_DB_PATH = RUNTIME_DIR / "sessions.sqlite"


@contextmanager
def open_session_checkpointer() -> Iterator[SqliteSaver]:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)

    with SqliteSaver.from_conn_string(
        str(CHECKPOINT_DB_PATH)
    ) as checkpointer:
        yield checkpointer