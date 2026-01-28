"""MIA Backend - WebSocket Stream Handler"""
from fastapi import APIRouter

websocket_router = APIRouter()

@websocket_router.get("/ws/status")
async def websocket_status():
    return {"status": "WebSocket endpoint available", "endpoint": "/ws/stream"}
