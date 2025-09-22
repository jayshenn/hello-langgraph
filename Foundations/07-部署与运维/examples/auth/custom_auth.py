# custom_auth.py
import jwt
import time
from typing import Dict, Any, Optional
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer

security = HTTPBearer()

class CustomAuthManager:
    """自定义认证管理器"""

    def __init__(self, secret_key: str):
        self.secret_key = secret_key

    def generate_token(self, user_id: str, roles: list = None) -> str:
        """生成JWT令牌"""
        payload = {
            "user_id": user_id,
            "roles": roles or [],
            "iat": time.time(),
            "exp": time.time() + 3600  # 1小时过期
        }
        return jwt.encode(payload, self.secret_key, algorithm="HS256")

    def validate_token(self, token: str) -> Dict[str, Any]:
        """验证JWT令牌"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=["HS256"])
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="令牌已过期")
        except jwt.InvalidTokenError:
            raise HTTPException(status_code=401, detail="无效的令牌")

# 认证依赖
auth_manager = CustomAuthManager("your-secret-key")

async def get_current_user(token: str = Depends(security)) -> Dict[str, Any]:
    """获取当前用户"""
    payload = auth_manager.validate_token(token.credentials)
    return {
        "user_id": payload["user_id"],
        "roles": payload.get("roles", [])
    }

def require_role(required_role: str):
    """要求特定角色的装饰器"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            current_user = kwargs.get("current_user")
            if not current_user or required_role not in current_user.get("roles", []):
                raise HTTPException(status_code=403, detail="权限不足")
            return await func(*args, **kwargs)
        return wrapper
    return decorator