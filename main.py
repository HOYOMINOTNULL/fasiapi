from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from api.facedata.face import app as facedata
from api.examination.examination import app as examination
from api.examination.record import app as record
from fastapi.staticfiles import StaticFiles


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(facedata)
app.include_router(examination)
app.include_router(record)

app.mount("/", StaticFiles(directory=r".\api\static", html=True), name="static")


if __name__ == '__main__':
    uvicorn.run(app="main:app", host='0.0.0.0', port=8000, workers=2)