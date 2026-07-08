import os
from pathlib import Path


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def get_data_root() -> Path:
    env_path = os.getenv("XANYLABELING_DATA_DIR")
    if env_path:
        return Path(env_path).expanduser().resolve()
    return get_project_root() / "xanylabeling_data"


def get_data_path(*parts: str) -> Path:
    return get_data_root().joinpath(*parts)
