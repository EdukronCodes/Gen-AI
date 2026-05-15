from fastapi import APIRouter

router = APIRouter(tags=["Health"])


@router.get("/health")
def health():
    return {"status": "healthy", "service": "retail-support-api"}


@router.get("/ready")
def ready():
    return {"status": "ready"}
