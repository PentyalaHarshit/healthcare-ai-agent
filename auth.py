from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer

SECRET_KEY = "change_this_secret_key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

# Use pbkdf2_sha256 to avoid bcrypt native dependency issues in some environments
pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")
# Set auto_error=False so we can handle missing tokens gracefully in optional flows
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/login", auto_error=False)


fake_users_db = {
    "admin": {
        "username": "admin",
        "password": pwd_context.hash("admin123"),
        "role": "admin"
    },
    "doctor": {
        "username": "doctor",
        "password": pwd_context.hash("doctor123"),
        "role": "doctor"
    },
    "patient": {
        "username": "patient",
        "password": pwd_context.hash("patient123"),
        "role": "patient"
    }
}


def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def authenticate_user(username: str, password: str):
    user = fake_users_db.get(username)

    if not user:
        return None

    if not verify_password(password, user["password"]):
        return None

    return user


def get_current_user(token: str = Depends(oauth2_scheme)):
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        role = payload.get("role")

        if username is None or role is None:
            raise HTTPException(status_code=401, detail="Invalid token")

        return {"username": username, "role": role}

    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")


def get_current_user_optional(token: str = Depends(oauth2_scheme)):
    """Return current user if token valid, otherwise return None (no exception).

    Useful for endpoints that allow anonymous access but can use user info
    when provided.
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        role = payload.get("role")

        if username is None or role is None:
            return None

        return {"username": username, "role": role}

    except Exception:
        return None


def require_roles(allowed_roles: list[str]):
    def checker(current_user=Depends(get_current_user)):
        if current_user["role"] not in allowed_roles:
            raise HTTPException(
                status_code=403,
                detail="Permission denied"
            )

        return current_user

    return checker  