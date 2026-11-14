FROM ubuntu:24.04
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates python3 python-is-python3 python3-numpy python3-scipy \
    && rm -rf /var/lib/apt/lists/*
COPY server/data/out/net.json server/data/out/zones.json server/data/catalogue.json /data/
WORKDIR /app
