from fastapi import Request
from app.services.mongodb_service import _get_db
from app.services.pinecone_service import _get_index


def get_compiled_graph(request: Request):
    return request.app.state.graph


def get_mongo_db():
    try:
        return _get_db()
    except Exception:
        return None


def get_pinecone_index():
    try:
        return _get_index()
    except Exception:
        return None
