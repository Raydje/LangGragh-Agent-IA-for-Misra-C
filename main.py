from fastapi import FastAPI
from app.api.routes import router
from app.config import get_settings

app = FastAPI(
    title="DO-178B Compliance Agent",
    description="Autonomous regulatory compliance analysis using LangGraph multi-agent system",
    version="0.1.0",
)

app.include_router(router, prefix="/api/v1")


@app.on_event("startup")
async def startup():
    settings = get_settings()
    print(f"[Startup] Loaded config for standard: DO-178B")
    print(f"[Startup] Gemini model: {settings.gemini_model}")
    print(f"[Startup] MongoDB: {settings.mongodb_uri}/{settings.mongodb_database}")
    print(f"[Startup] Pinecone index: {settings.pinecone_index_name}")
