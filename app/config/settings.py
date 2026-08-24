from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    DATABASE_URL: str
    OPENAI_API_KEY: str = ""
    QDRANT_API_KEY: str = ""
    QDRANT_CLUSTER_ENDPOINT: str = ""
    EMBEDDING_MODEL: str = "text-embedding-3-large"
    CHAT_MODEL: str = "gpt-4o-mini"
    COLLECTION_NAME: str = "quran_ayahs"
    SIMILARITY_THRESHOLD: float = 0.40

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
