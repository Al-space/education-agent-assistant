from fastapi import FastAPI
from app.services.config import settings
from app.api import kb_routes

app = FastAPI(title="教育智能题库助手", version="v1.0")
app.include_router(kb_routes.router)

@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "service": "education‑assistant",
        "version": "v1.0"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=settings.HOST, port=settings.PORT)