from langchain_google_genai import ChatGoogleGenerativeAI
from app.config import get_settings


def get_llm(temperature: float = 0.7) -> ChatGoogleGenerativeAI:
    settings = get_settings()
    return ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.gemini_api_key,
        temperature=temperature,
        convert_system_message_to_human=True,
    )
