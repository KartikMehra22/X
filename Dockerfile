FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (needed for some python packages if they build from source)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

# Bake the data and index into the image
RUN python scraper.py && python build_index.py

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
