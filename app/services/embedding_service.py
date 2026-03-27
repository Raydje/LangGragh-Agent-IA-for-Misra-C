from langchain_google_genai import GoogleGenerativeAIEmbeddings
from app.config import get_settings

_embeddings = None


def _get_embeddings_model() -> GoogleGenerativeAIEmbeddings:
    global _embeddings
    if _embeddings is None:
        settings = get_settings()
        _embeddings = GoogleGenerativeAIEmbeddings(
            model=settings.gemini_embedding_model,
            google_api_key=settings.gemini_api_key,
            task_type="retrieval_query",
        )
    return _embeddings


async def get_embedding(text: str) -> list[float]:
    model = _get_embeddings_model()
    return await model.aembed_query(text)


async def get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    model = _get_embeddings_model()
    return await model.aembed_documents(texts)
