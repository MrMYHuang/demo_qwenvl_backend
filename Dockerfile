# ---- Base image ----
FROM python:3.12-slim

# Install uv
RUN pip install --no-cache-dir uv

# Set working directory
WORKDIR /app

# Copy project files
#COPY pyproject.toml uv.lock* ./
COPY . .

# Install dependencies using uv
RUN uv sync --frozen --no-cache --no-dev

# Run your app
CMD ["uv", "run", "python", "server.py"]
