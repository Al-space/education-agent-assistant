from fastapi import FastAPI
from app.services.config import settings

app = FastAPI(title="智能题库查询测试",version="v1.0")

# 健康检查接口
@app.get("/health")
async def health_check():
    return {
        "status":"ok",
        "service":"education-assistant",
        "version":"v1.0"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("test_api:app",host=settings.HOST,port=settings.PORT)