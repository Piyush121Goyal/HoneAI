"""Hone AI — FastAPI application entrypoint."""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .database import init_db
from .routers import auth, health, optimize, prompts

if settings.sentry_dsn:
    import sentry_sdk

    sentry_sdk.init(dsn=settings.sentry_dsn, traces_sample_rate=0.1, environment=settings.environment)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield


app = FastAPI(
    title="Hone AI API",
    version="1.0.0",
    description="Turn a rough idea into a production-grade, structured prompt.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,  # needed for the httpOnly refresh cookie
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(auth.router)
app.include_router(optimize.router)
app.include_router(prompts.router)


@app.get("/", tags=["root"])
async def root():
    return {"name": settings.app_name, "docs": "/docs"}
