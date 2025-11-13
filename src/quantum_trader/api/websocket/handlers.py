"""WebSocket message handlers."""
from typing import Dict, Any
from fastapi import WebSocket
from structlog import get_logger

logger = get_logger(__name__)

class WebSocketMessageHandler:
    def __init__(self, config: Dict):
        self.config = config
        
    async def handle_message(self, websocket: WebSocket, message: Dict[str, Any]):
        """Handle incoming WebSocket message."""
        message_type = message.get("type")
        
        if message_type == "subscribe":
            await self.handle_subscribe(websocket, message)
        elif message_type == "unsubscribe":
            await self.handle_unsubscribe(websocket, message)
        elif message_type == "ping":
            await self.handle_ping(websocket)
        else:
            logger.warning("Unknown message type", type=message_type)
            
    async def handle_subscribe(self, websocket: WebSocket, message: Dict):
        """Handle subscription request."""
        channel = message.get("channel")
        symbol = message.get("symbol")
        
        logger.info("Subscribe request", channel=channel, symbol=symbol)
        
        await websocket.send_json({
            "type": "subscribed",
            "channel": channel,
            "symbol": symbol
        })
        
    async def handle_unsubscribe(self, websocket: WebSocket, message: Dict):
        """Handle unsubscription request."""
        channel = message.get("channel")
        
        logger.info("Unsubscribe request", channel=channel)
        
        await websocket.send_json({
            "type": "unsubscribed",
            "channel": channel
        })
        
    async def handle_ping(self, websocket: WebSocket):
        """Handle ping message."""
        await websocket.send_json({"type": "pong"})
