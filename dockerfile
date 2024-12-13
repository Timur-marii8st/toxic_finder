FROM python:3.13.1

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

RUN python -m nltk.downloader stopwords

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
