import logging
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
import base64
from io import BytesIO
from PIL import Image as PILImage  


log_directory = "./logs"
if not os.path.exists(log_directory):
    os.makedirs(log_directory)


log_file = os.path.join(log_directory, "serving.log")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', filename=log_file, filemode='w')
logger = logging.getLogger(__name__)


try:
    MODEL = tf.keras.models.load_model('/Users/ignaciocarrenoromero/PROYECTOS PERSONALES/MALARIA-DETECTOR/serving_project/model/model_epoch_20.keras')
    logger.info("Modelo cargado exitosamente")
except Exception as e:
    logger.error(f"Error al cargar el modelo: {str(e)}")
    raise

app = FastAPI()

@app.get("/")
def read_root():
    logger.info("Root endpoint was accessed")
    return {"Hello": "World"}

class Image(BaseModel):
    image: str

def preprocess_image(image: PILImage.Image) -> np.ndarray:
    logger.info("Preprocessing image")
    image = image.resize((150, 150))
    image = np.array(image) / 255.0
    if image.shape[-1] == 4:  # remove alpha channel
        image = image[..., :3]
    image = np.expand_dims(image, axis=0)  # add batch dimension
    return image

@app.post("/predict/")
def predict(image: Image):
    try:
        logger.info("Prediction endpoint was accessed")
        image_data = base64.b64decode(image.image)
        image = PILImage.open(BytesIO(image_data))  
        processed_image = preprocess_image(image)
        prediction = MODEL.predict(processed_image)
        predict_class = np.argmax(prediction, axis=1)[0]  #
        logger.info(f"Prediction made: {predict_class}")
        return {"prediction": int(predict_class)}
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)