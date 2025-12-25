FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      ca-certificates \
      curl \
      fontconfig \
      libjpeg62-turbo \
      libx11-6 \
      libxext6 \
      libxrender1 \
      libssl3 \
      xfonts-75dpi \
      xfonts-base \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN sed -E 's/ --hash=[^ ]+//g' requirements.txt > requirements_nohash.txt && \
    pip install --no-cache-dir -r requirements_nohash.txt && \
    rm requirements_nohash.txt

RUN pip install --no-cache-dir playwright && \
    playwright install --with-deps


COPY . .
ENV PYTHONPATH=/app

CMD ["python", "-m", "bots.telegram.app"]
