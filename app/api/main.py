"""FastAPI application entry point."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import auth, digest, search, sources
from app.core.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    print("Starting Social Media Radar API...")
    yield
    # Shutdown
    print("Shutting down Social Media Radar API...")


app = FastAPI(
    title="Social Media Radar API",
    description="Multi-channel intelligence aggregation system",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router, prefix="/api/v1/auth", tags=["Authentication"])
app.include_router(sources.router, prefix="/api/v1/sources", tags=["Sources"])
app.include_router(digest.router, prefix="/api/v1/digest", tags=["Digest"])
app.include_router(search.router, prefix="/api/v1/search", tags=["Search"])


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "Social Media Radar API",
        "version": "0.1.0",
        "status": "operational",
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}

