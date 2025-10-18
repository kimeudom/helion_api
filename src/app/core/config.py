from typing import Tuple

try:
    import pydantic as _pydantic  # type: ignore
except Exception:
    _pydantic = None

_pydantic_version: Tuple[int, ...]
if _pydantic is None:
    _pydantic_version = (0,)
else:
    ver = getattr(_pydantic, "__version__", "0")
    _pydantic_version = tuple(int(p)
                              for p in ver.split(".") if p.isdigit()) or (0,)

# Prefer pydantic v1 (BaseSettings) when available, otherwise require pydantic-settings
if _pydantic_version and _pydantic_version[0] >= 2:
    # pydantic v2 detected
    try:
        from pydantic_settings import BaseSettings  # type: ignore
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "pydantic v2 detected but `pydantic-settings` is not installed. "
            "Install it with: pip install pydantic-settings"
        ) from exc
    _USE_PYDANTIC_SETTINGS = True
else:
    # pydantic v1 (or not installed) — fall back to classic BaseSettings
    try:
        from pydantic import BaseSettings  # type: ignore
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Neither pydantic v1 nor pydantic-settings (for v2) is available. "
            "Install one of them: pip install 'pydantic<2' or pip install pydantic pydantic-settings"
        ) from exc
    _USE_PYDANTIC_SETTINGS = False


if _USE_PYDANTIC_SETTINGS:
    # pydantic-settings API (v2)
    class Settings(BaseSettings):
        app_name: str = "helion_api"
        version: str = "0.1.0"
        debug: bool = False

        # data and model paths (override via .env)
        data_geojson: str = "data/areas.geojson"
        model1_path: str = ""  # socio-economic need model (joblib)
        model3a_path: str = ""  # dynamic cost model (joblib)
        max_suggestions_default: int = 10

        model_config = {
            "env_file": ".env",
            "env_file_encoding": "utf-8",
        }

else:
    # Classic pydantic v1 BaseSettings
    class Settings(BaseSettings):
        app_name: str = "helion_api"
        version: str = "0.1.0"
        debug: bool = False

        # data and model paths (override via .env)
        data_geojson: str = "data/areas.geojson"
        model1_path: str = ""  # socio-economic need model (joblib)
        model3a_path: str = ""  # dynamic cost model (joblib)
        max_suggestions_default: int = 10

        class Config:  # type: ignore
            env_file = ".env"
            env_file_encoding = "utf-8"
