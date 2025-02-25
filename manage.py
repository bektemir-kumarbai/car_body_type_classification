from fastapi import FastAPI
from web.media_handler.routers import media_router
from config import app_port


app = FastAPI()
app.include_router(media_router)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(app_port))
