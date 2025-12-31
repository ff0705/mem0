import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional

import yaml
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, APIRouter, Depends, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from starlette.requests import Request
import uvicorn

from mem0 import AsyncMemory  # 改用异步版本

# 配置日志
DEBUG = os.getenv("DEBUG", "") in ["1", "true"]
logging.basicConfig(level=logging.DEBUG if DEBUG else logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

# 加载环境变量
load_dotenv()

# 安全配置
API_KEY = os.getenv("API_KEY", "")

# 数据库配置（从环境变量或默认值读取）
POSTGRES_HOST = os.environ.get("POSTGRES_HOST", "postgres")
POSTGRES_PORT = os.environ.get("POSTGRES_PORT", "5432")
POSTGRES_DB = os.environ.get("POSTGRES_DB", "postgres")
POSTGRES_USER = os.environ.get("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "postgres")
POSTGRES_COLLECTION_NAME = os.environ.get("POSTGRES_COLLECTION_NAME", "memories")

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://neo4j:7687")
NEO4J_USERNAME = os.environ.get("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "mem0graph")

# LLM和嵌入模型配置（兼容原文件的DeepSeek和SiliconFlow）
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "sk-0366528ec02848099d8a897048a93e3a")
EMBEDDER_API_KEY = os.environ.get("EMBEDDER_API_KEY", "sk-nucpbxyydswpwjsgkclqsdtejqtbhhkrxiqayvfiizfjuhyo")
HISTORY_DB_PATH = os.environ.get("HISTORY_DB_PATH", "./history/history.db")

# 默认配置（整合原文件的向量存储和模型配置）
DEFAULT_CONFIG = {
    "version": "v1.1",
    "vector_store": {
        "provider": "qdrant",  # 保留原文件的Qdrant配置
        "config": {
            "url": os.environ.get("QDRANT_URL", "http://192.168.1.101:6333"),
            "collection_name": os.environ.get("QDRANT_COLLECTION", "mem0"),
            "embedding_model_dims": 1024
        }
    },
    "graph_store": {
        "provider": "neo4j",
        "config": {"url": NEO4J_URI, "username": NEO4J_USERNAME, "password": NEO4J_PASSWORD}
    },
    "llm": {
        "provider": "openai",
        "config": {
            "api_key": OPENAI_API_KEY,
            "model": os.environ.get("LLM_MODEL", "deepseek-chat"),
            "openai_base_url": os.environ.get("LLM_BASE_URL", "https://api.deepseek.com/v1"),
            "temperature": 0.1,
            "max_tokens": 2000
        }
    },
    "embedder": {
        "provider": "openai",
        "config": {
            "api_key": EMBEDDER_API_KEY,
            "model": os.environ.get("EMBEDDER_MODEL", "BAAI/bge-m3"),
            "openai_base_url": os.environ.get("EMBEDDER_BASE_URL", "https://api.siliconflow.cn/v1")
        }
    },
    "history_db_path": HISTORY_DB_PATH,
}

# 从配置文件加载配置（如果存在）
config_path = os.environ.get("CONFIG_PATH", "./config.yaml")
if os.path.exists(config_path):
    with open(config_path, "r", encoding="utf-8") as rf:
        DEFAULT_CONFIG = yaml.safe_load(rf)

# 处理自定义提示中的时间变量
custom_fact_extraction_prompt = DEFAULT_CONFIG.get("custom_fact_extraction_prompt")
if custom_fact_extraction_prompt is not None:
    DEFAULT_CONFIG["custom_fact_extraction_prompt"] = custom_fact_extraction_prompt.replace(
        "${ENV_CUR_TIME}",
        datetime.now().strftime("%Y-%m-%d")
    )
logging.info(f"最终配置: {json.dumps(DEFAULT_CONFIG, indent=4, ensure_ascii=False)}")


# 异步上下文管理器初始化内存实例
memory_lock = asyncio.Lock()

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        async with memory_lock:
            app.state.memory = await AsyncMemory.from_config(DEFAULT_CONFIG)  # 异步初始化
        yield
    except Exception as e:
        logging.exception(f"初始化失败: {e}")


# 依赖项：获取内存实例
def get_memory(request: Request) -> AsyncMemory:
    return request.app.state.memory


# API密钥验证
api_key_header = APIKeyHeader(name="Authorization", auto_error=False)

def verify_api_key(authorization: Optional[str] = Security(api_key_header)):
    if API_KEY != "":
        if not authorization or not authorization.startswith("Token "):
            raise HTTPException(status_code=401, detail="未授权访问")
        token = authorization.replace("Token ", "").strip()
        if token != API_KEY:
            raise HTTPException(status_code=403, detail="密钥无效")


# 数据模型
class Message(BaseModel):
    role: str = Field(..., description="消息角色（user或assistant）")
    content: str = Field(..., description="消息内容")

class MemoryCreate(BaseModel):
    messages: List[Message] = Field(..., description="待存储的消息列表")
    user_id: Optional[str] = None
    app_id: Optional[str] = None
    run_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    infer: bool = True
    memory_type: Optional[str] = None
    prompt: Optional[str] = None

class SearchRequest(BaseModel):
    query: str = Field(..., description="搜索查询语句")
    user_id: Optional[str] = None
    app_id: Optional[str] = None
    run_id: Optional[str] = None
    limit: Optional[int] = 10
    threshold: Optional[float] = None
    filters: Optional[Dict[str, Any]] = None


# 初始化FastAPI应用
app = FastAPI(
    title="Mem0 API (集成异步增强版)",
    description="基于Mem0的记忆管理API，支持异步操作和扩展功能",
    version="2.0.0",
    lifespan=lifespan
)

# 配置CORS（保留原文件的宽松策略，生产环境建议收紧）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 路由配置（使用API密钥验证）
router = APIRouter(dependencies=[Depends(verify_api_key)])


# 保留原文件的基础接口
@router.get("/v1/ping/")
async def ping():
    return {
        "status": "success",
        "user_email": "wewins@we-wins.com",
        "message": "pong"
    }

@router.get("/health")
async def health_check():
    return {"status": "healthy", "port": os.getenv("PORT", 7888)}


# 新增异步增强接口
@router.post("/v1/memories/")
async def add_memory(
    memory_create: MemoryCreate,
    memory: AsyncMemory = Depends(get_memory)
):
    if not any([memory_create.user_id]):
        raise HTTPException(status_code=400, detail="请传入user_id参数")
    try:
        params = {k: v for k, v in memory_create.model_dump().items() if v is not None and k != "messages"}
        response = await memory.add(messages=[m.model_dump() for m in memory_create.messages], **params)
        return JSONResponse(content=response)
    except Exception as e:
        logging.exception("添加记忆失败")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/v1/memories/")
async def get_memories(
    user_id: Optional[str] = None,
    app_id: Optional[str] = None,
    run_id: Optional[str] = None,
    memory: AsyncMemory = Depends(get_memory)
):
    if not any([user_id]):
        raise HTTPException(status_code=400, detail="请传入user_id参数")
    try:
        params = {k: v for k, v in {"user_id": user_id, "app_id": app_id, "run_id": run_id}.items() if v is not None}
        return await memory.get_all(** params)
    except Exception as e:
        logging.exception("获取记忆失败")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/v1/memories/search/")
async def search_memories(
    search_req: SearchRequest,
    memory: AsyncMemory = Depends(get_memory)
):
    try:
        params = {k: v for k, v in search_req.model_dump().items() if v is not None and k != "query"}
        return await memory.search(query=search_req.query, **params)
    except Exception as e:
        logging.exception("搜索记忆失败")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/v1/memories/{memory_id}/")
async def get_memory(memory_id: str, memory: AsyncMemory = Depends(get_memory)):
    try:
        return await memory.get(memory_id)
    except Exception as e:
        logging.exception(f"获取记忆 {memory_id} 失败")
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/v1/memories/{memory_id}/")
async def update_memory(
    memory_id: str,
    updated_data: Dict[str, Any],
    memory: AsyncMemory = Depends(get_memory)
):
    try:
        return await memory.update(memory_id=memory_id, data=updated_data)
    except Exception as e:
        logging.exception(f"更新记忆 {memory_id} 失败")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/v1/memories/{memory_id}/")
async def delete_memory(memory_id: str, memory: AsyncMemory = Depends(get_memory)):
    try:
        await memory.delete(memory_id=memory_id)
        return {"message": "记忆删除成功"}
    except Exception as e:
        logging.exception(f"删除记忆 {memory_id} 失败")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/v1/memories/")
async def delete_all_memories(
    user_id: Optional[str] = None,
    app_id: Optional[str] = None,
    run_id: Optional[str] = None,
    memory: AsyncMemory = Depends(get_memory)
):
    if not any([user_id, app_id, run_id]):
        raise HTTPException(status_code=400, detail="至少需要一个标识符")
    try:
        params = {k: v for k, v in {"user_id": user_id, "app_id": app_id, "run_id": run_id}.items() if v is not None}
        await memory.delete_all(** params)
        return {"message": "所有相关记忆已删除"}
    except Exception as e:
        logging.exception("批量删除记忆失败")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/v1/configure")
async def configure_memory(config: Dict[str, Any], request: Request):
    async with memory_lock:
        request.app.state.memory = await AsyncMemory.from_config(config)
    return {"message": "配置更新成功"}

@router.get("/")
async def home():
    return RedirectResponse(url="/docs")


# 注册路由
app.include_router(router)

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", 7888)),  # 支持环境变量配置端口
        log_level="debug" if DEBUG else "info"
    )