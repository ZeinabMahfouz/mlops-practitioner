FROM python:3.10-slim

WORKDIR /app
# load dependencies
COPY pyproject.toml .
COPY src/ src/
COPY models/ models/

#  install dependencies 
RUN pip install --no-cache-dir .

EXPOSE 8000

CMD ["uvicorn", "prodml.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
