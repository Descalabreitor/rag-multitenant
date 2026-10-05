# Image for the ragmt package. compose.yaml runs it as the permsync service
# (`python -m ragmt.permsync`); configuration comes from the environment only.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app
# Only what the wheel needs: no .env, tests or local data end up in the image.
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install .

USER nobody
CMD ["python", "-m", "ragmt.permsync"]
