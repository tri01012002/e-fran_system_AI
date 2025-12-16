"""
Lux Authentication API
Main application entry point
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn


# Import routers
from test_raiki_sdk import check_router, classify_router, brand_detect_router  


# Import registry to check registered models
from raiki_sdk import list_models  


# Create FastAPI app
app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Lifecycle events
@app.on_event("startup")
async def startup_event():
    """
    Run on application startup
    Display registered models
    """
    print("\n" + "="*60)
    print("🚀 Lux Authentication API Starting...")
    print("="*60)
    
    # Show registered models
    models = list_models()
    print(f"\n📦 Registered Models: {len(models)}")
    print("-"*60)
    
    for key, info in models.items():
        print(f"  ✅ {info['category'].upper()} {info['part']} - {info['version']}")
        print(f"     Package: {info['package_name']}")
    
    print("\n" + "="*60)
    print("✨ API is ready to accept requests!")
    print("="*60 + "\n")


@app.on_event("shutdown")
async def shutdown_event():
    """
    Run on application shutdown
    """
    print("\n👋 Shutting down Lux Authentication API...")


# Health check endpoint
@app.get("/", tags=["Health"])
async def root():
    """
    Root endpoint - API health check
    """
    return JSONResponse(
        status_code=200,
        content={
            "status": "ok",
            "message": "Lux Authentication API is running",
            "version": "1.0.0",
            "models_registered": len(list_models())
        }
    )


@app.get("/health", tags=["Health"])
async def health_check():
    """
    Health check endpoint
    """
    return JSONResponse(
        status_code=200,
        content={
            "status": "healthy",
            "models": {
                key: {
                    "category": info['category'],
                    "part": info['part'],
                    "version": info['version']
                }
                for key, info in list_models().items()
            }
        }
    )


# Include routers
app.include_router(
    check_router,
    prefix="/api/v1/check",
    tags=["Authentication"]
)

app.include_router(
    classify_router,
    prefix="/api/v1/classify",
    tags=["Classification"]
)

app.include_router(
    brand_detect_router,
    prefix="/api/v1/brand-detect",
    tags=["Brand Detection"]
)
# Run with uvicorn
if __name__ == "__main__":
    uvicorn.run(
        "test_main:app",
        host="0.0.0.0",
        port=8001,
        reload=False,
        log_level="info"
    )