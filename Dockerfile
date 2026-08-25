FROM python:3.10-slim

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# 1. Create a non-root group and user
RUN addgroup --system appgroup && adduser --system --group appuser

# 2. Copy dependencies and source code
COPY pyproject.toml .
COPY src/ src/
COPY models/ models/

# 3. Install dependencies and application
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

# 4. Change ownership of files to non-root user
RUN chown -R appuser:appgroup /app

# 5. Switch to non-root user (Fulfills DoD requirement)
USER appuser

EXPOSE 8000

CMD ["uvicorn", "prodml.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
