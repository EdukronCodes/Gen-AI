from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.config.settings import get_settings
from app.config.logging_config import setup_logging
from app.config.database import Base, engine
from app.database.seed import seed
from app.api.routes import customer_routes, order_routes, product_routes, refund_routes, admin_routes, auth_routes
from app.monitoring.healthcheck import router as health_router

settings = get_settings()
setup_logging(settings.debug)


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    seed()
    yield


app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(customer_routes.router)
app.include_router(order_routes.router)
app.include_router(product_routes.router)
app.include_router(refund_routes.router)
app.include_router(admin_routes.router)
app.include_router(auth_routes.router)


STATIC_DIR = Path(__file__).parent / "static"


@app.get("/")
def root():
    return {
        "service": settings.app_name,
        "docs": "/docs",
        "chat_ui": "/chat",
        "chat_api": "/api/v1/customer/chat",
    }


@app.get("/chat")
def chat_ui():
    return FileResponse(STATIC_DIR / "chat.html")


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
