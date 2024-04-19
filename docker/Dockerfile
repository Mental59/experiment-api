FROM python:3.10.9-slim

WORKDIR /usr/src/app

COPY requirements.txt requirements.txt

RUN apt-get update
RUN apt-get install libpq-dev gcc -y

RUN pip install psycopg2-binary --no-cache-dir
RUN pip install -r requirements.txt --no-cache-dir
RUN pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118 --no-cache-dir

COPY . .

EXPOSE 3333

CMD uvicorn app.main:app --port 3333 --host 0.0.0.0
