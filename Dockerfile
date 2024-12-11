FROM python:3.12

WORKDIR /app

COPY .streamlit/ /app/.streamlit/

COPY app.py poetry.toml poetry.lock pyproject.toml README.md /app/

RUN pip install poetry && poetry install

EXPOSE 8501

ENTRYPOINT ["poetry", "run", "streamlit", "run", "app.py"]
