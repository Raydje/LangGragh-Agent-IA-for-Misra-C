from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from app.api.routes import router
from app.config import get_settings
from app.utils import logger
from app.graph.builder import build_graph
import aiosqlite


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Logs configuration on startup to ensure environment variables are loaded."""
    settings = get_settings()
    logger.info(f"[Startup] Loaded config for standard: MISRA C:2023")
    logger.info(f"[Startup] Gemini model: {settings.gemini_model}")
    logger.info(f"[Startup] MongoDB: {settings.mongodb_uri}/{settings.mongodb_database}")
    logger.info(f"[Startup] Pinecone index: {settings.pinecone_index_name}")
    conn = await aiosqlite.connect("checkpoints.sqlite")
    app.state.graph = await build_graph(conn)
    yield
    await conn.close()


# Initialize FastAPI app with metadata for Swagger UI
app = FastAPI(
    title="MISRA C:2023 Compliance Agent",
    description="Autonomous regulatory compliance analysis using a LangGraph multi-agent system.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"status_code": exc.status_code, "error": exc.__class__.__name__, "detail": exc.detail},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception on {request.method} {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"status_code": 500, "error": "InternalServerError", "detail": "An unexpected internal server error occurred."},
    )

# Include the routes defined in routes.py
app.include_router(router, prefix="/api/v1")