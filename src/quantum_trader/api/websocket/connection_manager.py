"""WebSocket connection management."""
from typing import Dict, List, Set
from fastapi import WebSocket
from structlog import get_logger

logger = get_logger(__name__)

class WebSocketConnectionManager:
    def __init__(self):
        self.connections: Dict[str, Set[WebSocket]] = {}
        
    async def connect(self, websocket: WebSocket, client_id: str):
        """Connect a WebSocket client."""
        await websocket.accept()
        
        if client_id not in self.connections:
            self.connections[client_id] = set()
            
        self.connections[client_id].add(websocket)
        logger.info("Client connected", client_id=client_id)
        
    def disconnect(self, websocket: WebSocket, client_id: str):
        """Disconnect a WebSocket client."""
        if client_id in self.connections:
            self.connections[client_id].discard(websocket)
            
            if not self.connections[client_id]:
                del self.connections[client_id]
                
        logger.info("Client disconnected", client_id=client_id)
        
    async def send_personal_message(self, message: dict, client_id: str):
        """Send message to specific client."""
        if client_id in self.connections:
            for connection in self.connections[client_id]:
                try:
                    await connection.send_json(message)
                except Exception as e:
                    logger.error("Send failed", error=str(e), client_id=client_id)
                    
    async def broadcast_to_room(self, message: dict, room: str):
        """Broadcast to all clients in a room."""
        for client_id, connections in self.connections.items():
            if room in client_id:  # Simple room matching
                for connection in connections:
                    try:
                        await connection.send_json(message)
                    except Exception as e:
                        logger.error("Broadcast failed", error=str(e))
