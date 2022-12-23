# Get the Fast API image with Python version 3.7
FROM tiangolo/uvicorn-gunicorn-fastapi:python3.10

# Create the directory for the container
WORKDIR /app

COPY pyproject.toml /app 

ENV PYTHONPATH=${PYTHONPATH}:${PWD} 

RUN pip3 install poetry

RUN poetry config virtualenvs.create false

RUN poetry install --no-dev

COPY ./app.py ./

# Copy the serialized model and the vectors
COPY ./models/logistic_regression_hyper_tun.sav ./models/logistic_regression_hyper_tun.sav

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]