import uuid


def generate_correlation_id() -> str:
    return str(uuid.uuid4())
