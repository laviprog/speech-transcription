from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    LOG_LEVEL: str = "INFO"
    ENV: str = "prod"
    ROOT_PATH: str = "/speech-transcription/api/v1"

    DB_HOST: str
    DB_PORT: int
    DB_NAME: str
    DB_USER: str
    DB_PASSWORD: str

    SECRET_KEY: str
    SECRET_REFRESH_KEY: str
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 15
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    DEVICE: str = "cpu"
    COMPUTE_TYPE: str = "float32"
    DOWNLOAD_ROOT: str = "models"
    BATCH_SIZE: int = 4
    CHUNK_SIZE: int = 10

    ADMIN_USERNAME_DEFAULT: str = "admin"
    ADMIN_PASSWORD_DEFAULT: str = "password"

    @property
    def DB_URL(self) -> str:
        return f"postgresql+asyncpg://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"


settings = Settings()
