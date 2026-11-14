FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir numpy==2.5.4 scipy==1.18.1 numba==0.68.1 shapely==2.2.0 mcp==1.30.0
COPY sim /app/sim
COPY data /app/data
RUN python -c "from sim import City; City(0).forecast({}, 1)"
COPY test.sh /tests/
RUN mkdir -p /app/runs
