from fastapi import APIRouter  # type: ignore

router = APIRouter(tags=["v1"])


@router.get("/v1/hello", summary="Simple hello endpoint")
async def read_hello():
    return {"message": "Hello, Helion!"}
