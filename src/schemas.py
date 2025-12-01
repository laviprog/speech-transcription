from datetime import datetime, timezone

from pydantic import BaseModel, Field


class BaseSchema(BaseModel):
    model_config = {"from_attributes": True}


class HealthCheck(BaseModel):
    status: str = "ok"
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
