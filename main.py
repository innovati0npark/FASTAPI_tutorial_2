from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from PIL import Image, ImageEnhance
import io

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "Lion"}

@app.post("/upload/")
async def create_upload_file(file: UploadFile = File(...)):
    if file.content_type.startswith('image/'):
        # 이미지 파일 읽기
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))

        # 이미지를 그레이스케일로 변환
        gray_image = image.convert('L')

        # 변환된 이미지를 byte로 변환
        img_byte_arr = io.BytesIO()
        gray_image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()

        # StreamingResponse로 이미지 반환
        return StreamingResponse(io.BytesIO(img_byte_arr), media_type="image/png")
    else:
        raise HTTPException(status_code=400, detail="Invalid file format.")
    
@app.post("/rotate/")
async def rotate_image(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Invalid file format.")
    
    # 이미지 파일 읽기
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))

    # 이미지를 90도 회전
    rotated_image = image.rotate(90, expand=True)

    # 변환된 이미지를 byte로 변환
    img_byte_arr = io.BytesIO()
    rotated_image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    # StreamingResponse로 이미지 반환
    return StreamingResponse(io.BytesIO(img_byte_arr), media_type="image/png")

@app.post("/adjust_brightness/")
async def adjust_brightness(file: UploadFile = File(...), brightness_factor: float = 1.0):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Invalid file format.")
    
    # 이미지 파일 읽기
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))

    # 이미지의 밝기를 조절
    enhancer = ImageEnhance.Brightness(image)
    enhanced_image = enhancer.enhance(brightness_factor)

    # 변환된 이미지를 byte로 변환
    img_byte_arr = io.BytesIO()
    enhanced_image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    # StreamingResponse로 이미지 반환
    return StreamingResponse(io.BytesIO(img_byte_arr), media_type="image/png")


@app.post("/black_white/")
async def apply_black_white_filter(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Invalid file format.")
    
    # 이미지 파일 읽기
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))

    # 이미지를 그레이스케일로 변환
    bw_image = image.convert('L')

    # 변환된 이미지를 byte로 변환
    img_byte_arr = io.BytesIO()
    bw_image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    # StreamingResponse로 이미지 반환
    return StreamingResponse(io.BytesIO(img_byte_arr), media_type="image/png")


#붓꽃 분류 모델 Iris
from pydantic import BaseModel
import numpy as np
import pickle

# 모델 로드
with open("iris_model.pkl", "rb") as f:
    model = pickle.load(f)

class IrisModel(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.post("/predict")
def predict_iris(iris: IrisModel):
    data = np.array([[iris.sepal_length, iris.sepal_width, iris.petal_length, iris.petal_width]])
    prediction = model.predict(data)
    return {"prediction": int(prediction[0])}




from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pickle

# 모델 로드
with open("wine_model.pkl", "rb") as f:
    model = pickle.load(f)

class WineFeatures(BaseModel):
    features: list

@app.post("/predict_wine/")
def predict_wine(wine: WineFeatures):
    try:
        prediction = model.predict([wine.features])
        return {"prediction": int(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


