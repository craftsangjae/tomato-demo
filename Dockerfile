FROM python:3.12

WORKDIR /app

COPY .streamlit/ /app/.streamlit/
COPY src/ /app/src/

# TODO - 추론을 serving환경에서 작업하도록 변경한 후 제거
COPY data/ /app/data/

COPY app.py poetry.lock pyproject.toml README.md /app/

RUN pip install poetry && poetry install

EXPOSE 8501

ENTRYPOINT ["poetry", "run", "streamlit", "run", "app.py"]
