from fastapi import FastAPI  # type: ignore
from .core.config import Settings
from .api.v1 import hello, inference, predict

settings = Settings()

app = FastAPI(title=settings.app_name, version=settings.version)

# include versioned routers under /api
app.include_router(hello, prefix="/api")
app.include_router(inference, prefix="/api")
app.include_router(predict, prefix="/api")


@app.get("/", summary="App root")
def root():
    return {"app": settings.app_name, "version": settings.version}
