from fastapi import Request


def get_compiled_graph(request: Request):
    return request.app.state.graph


def get_mongo_db(request: Request):
    return request.app.state.mongodb.db


def get_pinecone_index(request: Request):
    return request.app.state.pinecone.index
