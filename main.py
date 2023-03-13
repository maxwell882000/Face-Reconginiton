import base64
from io import BytesIO
import httpx

from fastapi import FastAPI
import face_recognition
from starlette.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Body(BaseModel):
    compare: str
    unknown: str


@app.post("/model/image-comparison")
async def root(data: Body):
    # return {"detected" : True}
    async with httpx.AsyncClient() as client:
        object = await client.get(data.compare)
    file_like = BytesIO(base64.b64decode(data.unknown.replace("data:image/octet-stream;base64,", "")))
    compare_face = face_recognition.load_image_file(BytesIO(object.content))
    unknown_face = face_recognition.load_image_file(file_like)
    try:
        compare_encoding = face_recognition.face_encodings(compare_face)[0]
        unknown_encoding = face_recognition.face_encodings(unknown_face)
        if len(unknown_encoding) > 1 or len(unknown_encoding) == 0: 
            return {"detected" : False}
    except IndexError:
        return {"detected": False, "status": "Error"}
    e = face_recognition.compare_faces([compare_encoding], unknown_encoding[0])
    s = e[0]
    return {"detected": int(s)}
