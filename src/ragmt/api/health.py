"""Liveness probe. No auth and no database: it says the process is up, nothing more."""

from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(tags=["health"])


class Health(BaseModel):
    status: Literal["ok"] = "ok"


@router.get("/healthz")
async def healthz() -> Health:
    return Health()
