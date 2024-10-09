# Malaria Detection API

This project provides an API for detecting malaria in blood cell images using a deep learning model built with TensorFlow and FastAPI. The API allows you to send an image in base64 format to get a prediction of whether malaria is present.

## Project Structure

```
├── README.md
├── logs
│   └── serving.log
├── model
│   └── model_epoch_20.keras
├── requirements.txt
├── src
│   ├── __init__.py
│   └── app.py
└── tests
    └── test_app.py
```

- `logs/`: Directory where the logs for the API are stored.
- `model/`: Contains the trained machine learning model used for predictions.
- `src/`: The main source code directory containing the FastAPI app.
- `tests/`: Contains the unit tests for the API.
- `requirements.txt`: Python dependencies for the project.

## Prerequisites

- Python 3.12 or higher
- `pip` (Python package manager)
- Docker (optional, for containerized deployment)

## Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/IgnaCodeIA/Malaria-Detector-Serving
   cd Malaria-Detector-API
   ```

2. **Install dependencies:** Make sure you have a virtual environment activated and install the required dependencies.
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables (optional):** You can set the path to the model file using the `MODEL_PATH` environment variable. If not set, the default path will be used:
   ```bash
   export MODEL_PATH=/path/to/your/model/file.keras
   ```

4. **Run the FastAPI application:** You can start the server with:
   ```bash
   uvicorn src.app:app --host 0.0.0.0 --port 8000
   ```
   The API will be accessible at [http://127.0.0.1:8000](http://127.0.0.1:8000).

## API Endpoints

### Health Check
- **Endpoint:** `/`
- **Method:** GET
- **Description:** Checks if the API is running.

**Example Request**
```bash
curl -X 'GET' 'http://127.0.0.1:8000/' -H 'accept: application/json'
```

**Example Response**
```json
{
  "status": "running"
}
```

### Predict Malaria
- **Endpoint:** `/predict/`
- **Method:** POST
- **Description:** Takes a base64-encoded image and predicts whether the blood cell contains malaria.

**Request Body**
```json
{
  "image": "<base64-encoded-string>"
}
```

**Example Request**
```bash
curl -X 'POST' 'http://127.0.0.1:8000/predict/' \
-H 'accept: application/json' \
-H 'Content-Type: application/json' \
-d '{"image": "<base64-encoded-string>"}'
```

**Example Response**
```json
{
  "prediction": 0
}
```
Where `0` indicates "No Malaria" and `1` indicates "Malaria Detected".

## Running Tests

1. **Install pytest if not already installed:**
   ```bash
   pip install pytest
   ```

2. **Run the tests:**
   ```bash
   pytest
   ```

3. **Run tests with detailed output:**
   ```bash
   pytest -v
   ```

4. **Check test coverage (optional):** To measure code coverage, you need to install `pytest-cov`:
   ```bash
   pip install pytest-cov
   ```

   Then, run:
   ```bash
   pytest --cov=src tests/
   ```

## Using Docker

### Building the Docker Image
1. **Build the Docker image:**
   ```bash
   docker build -t malaria-detector-api .
   ```

2. **Run the Docker container:**
   ```bash
   docker run -it --rm -p 8000:8000 -v $(pwd)/model:/app/model malaria-detector-api
   ```

   The API will be accessible at [http://127.0.0.1:8000](http://127.0.0.1:8000).

## MLOps Best Practices

- **Logging:** The API logs are stored in the `logs/` directory and can be used to monitor API usage and detect issues.
- **Environment Variables:** Use environment variables to configure the path of the model file and other configurations.
- **Dockerization:** The provided Dockerfile allows for containerized deployment, making it easier to run the API in different environments.

## Dependencies

The main dependencies are listed in the `requirements.txt` file:

```
fastapi==0.95.0
tensorflow==2.13.0
pillow==10.0.0
uvicorn==0.22.0
numpy==1.24.3
pytest==7.4.0
pytest-cov==4.1.0
```