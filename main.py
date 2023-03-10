import base64
from io import BytesIO
import httpx

from fastapi import FastAPI
import face_recognition
from starlette.middleware.cors import CORSMiddleware

app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/model/image-comparison")
async def root(compare: str, unknown: str):
    async with httpx.AsyncClient() as client:
        object = await client.get(compare)
    file_like = BytesIO(base64.b64decode(unknown.replace("data:image/octet-stream;base64,", "")))
    compare_face = face_recognition.load_image_file(BytesIO(object.content))
    unknown_face = face_recognition.load_image_file(file_like)
    try:
        compare_encoding = face_recognition.face_encodings(compare_face)[0]
        unknown_encoding = face_recognition.face_encodings(unknown_face)[0]
    except IndexError:
        return {"detected": False, "status": "Error"}
    e = face_recognition.compare_faces([compare_encoding], unknown_encoding)
    s = e[0]
    return {"detected": int(s)}
