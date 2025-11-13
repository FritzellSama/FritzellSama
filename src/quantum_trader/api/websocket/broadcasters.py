"""WebSocket broadcast management."""
from typing import List, Set
from fastapi import WebSocket
from structlog import get_logger

logger = get_logger(__name__)

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info("WebSocket connected", total=len(self.active_connections))
        
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info("WebSocket disconnected", total=len(self.active_connections))
        
    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients."""
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error("Broadcast failed", error=str(e))


class MarketDataBroadcaster:
    def __init__(self):
        self.manager = ConnectionManager()
        
    async def broadcast_price_update(self, symbol: str, price: str):
        """Broadcast price update."""
        await self.manager.broadcast({
            "type": "price_update",
            "symbol": symbol,
            "price": price
        })
        
    async def broadcast_trade(self, trade_data: dict):
        """Broadcast trade execution."""
        await self.manager.broadcast({
            "type": "trade",
            "data": trade_data
        })
