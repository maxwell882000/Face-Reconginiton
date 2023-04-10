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


class BodyMultiple(BaseModel):
    compare: list
    unknown: str


@app.post("/model/image-comparison/multiple")
async def comparison(data: BodyMultiple):
    file_like = BytesIO(base64.b64decode(
        data.unknown.replace("data:image/octet-stream;base64,", "").replace("data:image/jpeg;base64,", "")))
    unknown_face = face_recognition.load_image_file(file_like)

    try:
        # map to face encodings ,then take only images that have faces, take first face
        compare_encoding = list(map(lambda x: {"avatar": x["avatar"][0], "id": x["id"]},
                                    filter(lambda x: len(x["avatar"]) > 0,
                                           list(
                                        map(lambda x:
                                            {"avatar": face_recognition.face_encodings(
                                                face_recognition.load_image_file(x['avatar'])),
                                             "id": x["id"]
                                             },
                                            data.compare)
                                    )
        )
        )
        )
        unknown_encoding = face_recognition.face_encodings(unknown_face)
    except Exception:
        return {"detected": False, "status": "Error"}
    for items in compare_encoding:
        if face_recognition.compare_faces([items["avatar"]], unknown_encoding[0])[0]:
            return {"detected": items["id"]}
    return {"detected": 0}


@app.post("/model/image-comparison")
async def root(data: Body):
    # return {"detected" : True}
    async with httpx.AsyncClient() as client:
        object = await client.get(data.compare)
    file_like = BytesIO(base64.b64decode(
        data.unknown.replace("data:image/octet-stream;base64,", "")))
    compare_face = face_recognition.load_image_file(BytesIO(object.content))
    unknown_face = face_recognition.load_image_file(file_like)
    try:
        compare_encoding = face_recognition.face_encodings(compare_face)[0]
        unknown_encoding = face_recognition.face_encodings(unknown_face)
        if len(unknown_encoding) > 1 or len(unknown_encoding) == 0:
            return {"detected": False}
    except IndexError:
        return {"detected": False, "status": "Error"}
    e = face_recognition.compare_faces([compare_encoding], unknown_encoding[0])
    s = e[0]
    return {"detected": int(s)}
