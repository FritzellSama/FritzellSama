"""FastAPI application initialization."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from structlog import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="Quantum Trader AI API",
    version="1.0.0",
    description="Institutional Trading Platform API"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    logger.info("API starting up")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("API shutting down")


@app.get("/")
async def root():
    return {"message": "Quantum Trader AI API", "status": "running"}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
