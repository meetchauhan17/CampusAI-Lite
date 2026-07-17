import os
from typing import List, Optional, Any
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

# Explicitly load .env file
load_dotenv()

class Settings(BaseSettings):
    """
    Project-wide settings loaded from environment variables and .env file.
    All properties are typed and validated via Pydantic.
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # watsonx.ai (referred internally as "bob")
    BOBSHELL_API_KEY: Optional[str] = None
    BOB_MODEL: str = "ibm/granite-3-8b-instruct"
    BOB_PROJECT_ID: Optional[str] = None
    BOB_URL: Optional[str] = None

    # Other API keys
    GROQ_API_KEY: Optional[str] = None
    GEMINI_API_KEY: Optional[str] = None

    # Fallback configuration
    PRIMARY_PROVIDER: str = "bob"
    # Typed as Any to bypass pydantic-settings complex JSON decoding from environment variables
    FALLBACK_ORDER: Any = ["bob", "groq", "gemini"]

    @field_validator("FALLBACK_ORDER", mode="before")
    @classmethod
    def parse_fallback_order(cls, v: Any) -> List[str]:
        if isinstance(v, str):
            if v.startswith("[") and v.endswith("]"):
                try:
                    import json
                    return json.loads(v)
                except Exception:
                    pass
            return [item.strip() for item in v.split(",") if item.strip()]
        return v

    def validate_keys(self) -> None:
        """Helper to check if active keys are missing and warn the user."""
        missing = []
        if not self.BOBSHELL_API_KEY:
            missing.append("BOBSHELL_API_KEY")
        if not self.BOB_PROJECT_ID:
            missing.append("BOB_PROJECT_ID")
        if not self.GROQ_API_KEY:
            missing.append("GROQ_API_KEY")
        if not self.GEMINI_API_KEY:
            missing.append("GEMINI_API_KEY")
        
        if missing:
            # We don't have a logger imported here yet to avoid circular dependency,
            # so we just print a helper note.
            print(f"[Settings Warning] Missing environment variables: {', '.join(missing)}")
