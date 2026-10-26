import os
from dotenv import load_dotenv

# 加载项目根目录 .env 文件
load_dotenv()

class Settings:
    # 服务
    HOST: str = os.getenv("HOST","0.0.0.0")
    PORT: int = int(os.getenv("PORT",8000))

    # LLM SiliconFlow
    SILICONFLOW_API_KEY:str = os.getenv("SILICONFLOW_API_KEY","")
    SILICONFLOW_BASE_URL:str = os.getenv("SILICONFLOW_BASE_URL")

    # 工具密钥
    SERPAPI_API_KEY:str = os.getenv("SERPAPI_API_KEY","")
    SENIVERSE_API_KEY:str = os.getenv("SENIVERSE_API_KEY","")

settings = Settings()