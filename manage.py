from fastapi import FastAPI
from web.media_handler.routers import media_router
from decouple import config
APP_PORT = config("APP_PORT", cast=int, default=8080)

app = FastAPI()
app.include_router(media_router)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=APP_PORT)
