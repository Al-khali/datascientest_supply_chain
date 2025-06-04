from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, OAuth2PasswordBearer
from pydantic import BaseModel, validator, Field
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union
import logging
import os
import time
from jose import JWTError, jwt
from passlib.context import CryptContext

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Customer Satisfaction API",
    description="API for analyzing customer reviews",
    version="1.0.0",
    openapi_url="/api/v1/openapi.json"
)

security = HTTPBearer()

# Mock data models (to be replaced with real implementations)
class ReviewData(BaseModel):
    id: str
    author: str
    content: str
    rating: int
    date: datetime
    source: str

    @validator('rating')
    def validate_rating_range(cls, v, values):
        if v < 1 or v > 5:
            raise ValueError('Rating must be between 1 and 5')
        return v

class KPIResponse(BaseModel):
    avg_rating: float
    response_rate: float
    common_complaints: List[str]
    sentiment_score: float

class TrendAnalysis(BaseModel):
    period: str
    avg_rating: float
    complaint_count: int

class BusinessInsight(BaseModel):
    id: str
    title: str
    description: str
    impact: str
    priority: str

# Security configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "09d25e094faa6ca2556c818166b7a9563b93f7099f6f0f4caa6cf63b88e8d3e7")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

class TokenData(BaseModel):
    username: Union[str, None] = None

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        token = credentials.credentials
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    return token_data

# Middleware for processing time measurement
@app.middleware("http")
async def add_process_time_header(request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# Health check endpoint
@app.get("/health", summary="Service health check", response_model=Dict[str, str])
async def health_check():
    """Check service availability"""
    return {"status": "ok"}

# Metrics endpoint
@app.get("/metrics", summary="Service metrics", response_model=Dict[str, Any])
async def get_metrics():
    """Get service performance metrics"""
    # In production, integrate with Prometheus
    return {
        "uptime": time.time() - start_time,
        "requests_processed": 0,  # Would be dynamic in real implementation
        "error_rate": 0.0
    }

# API endpoints with improved error handling
@app.get("/api/v1/kpis", 
         response_model=KPIResponse,
         summary="Get customer satisfaction KPIs",
         responses={
             401: {"description": "Unauthorized"},
             500: {"description": "Internal server error"}
         })
async def get_advanced_kpis(token: TokenData = Depends(verify_token)):
    """
    Retrieve advanced customer satisfaction KPIs with caching and error handling.
    """
    try:
        # In real implementation, this would call application layer services
        return KPIResponse(
            avg_rating=4.2,
            response_rate=0.85,
            common_complaints=["Delivery", "Product Quality"],
            sentiment_score=0.78
        )
    except Exception as e:
        logger.error(f"Error fetching KPIs: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error processing request"
        )

@app.get("/api/v1/motifs", response_model=Dict[str, int])
async def get_motifs_distribution(token: str = Depends(verify_token)):
    """Distribution des motifs d'insatisfaction avec comptages."""
    # Mock implementation
    return {"Delivery": 120, "Product Quality": 85, "Customer Service": 45}

@app.get("/api/v1/trends", response_model=List[TrendAnalysis])
async def get_trends_analysis(token: str = Depends(verify_token)):
    """Analyse des tendances sur plusieurs périodes."""
    # Mock implementation
    trends = []
    periods = ["2023-01", "2023-02", "2023-03"]
    for p in periods:
        trend = TrendAnalysis(
            period=p,
            avg_rating=4.0 + (0.1 * periods.index(p)),
            complaint_count=100 - (10 * periods.index(p))
        )
        trends.append(trend)
    return trends

@app.get("/api/v1/insights", response_model=List[BusinessInsight])
async def get_business_insights(token: str = Depends(verify_token)):
    """
    Insights business avec priorisation basée sur l'impact.
    """
    # Mock implementation
    return [
        BusinessInsight(
            id="1",
            title="Amélioration des délais de livraison",
            description="Réduction des délais de livraison de 24h",
            impact="High",
            priority="Urgent"
        ),
        BusinessInsight(
            id="2",
            title="Formation du service client",
            description="Programme de formation sur la gestion des réclamations",
            impact="Medium",
            priority="High"
        )
    ]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
