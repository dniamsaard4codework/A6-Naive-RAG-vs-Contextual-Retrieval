FROM python:3.12-slim

WORKDIR /workspace

# Install uv for fast, reproducible installs
RUN pip install --no-cache-dir uv

# Install only the runtime packages needed by the app
# (avoids pulling large ML frameworks like torch into the container)
RUN uv pip install --system --no-cache \
    openai \
    langchain \
    langchain-openai \
    langchain-community \
    faiss-cpu \
    streamlit \
    python-dotenv \
    pymupdf \
    nest-asyncio

# Copy app source
COPY app/app.py app/rag.py /workspace/app/

WORKDIR /workspace/app

# The FAISS index and data are mounted as volumes at runtime (see docker-compose.yml)
# FAISS_INDEX_PATH tells rag.py where to find the pre-built contextual index
ENV FAISS_INDEX_PATH=/workspace/index/contextual_faiss

EXPOSE 8501

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
