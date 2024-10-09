FROM python:3.12

ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=/app/model/model_epoch_20.keras

WORKDIR /app

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

EXPOSE 8000

CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
