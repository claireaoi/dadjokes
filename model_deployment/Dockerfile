FROM "python:3.9"

WORKDIR /app
COPY requirements.txt ./
RUN pip3 install -r requirements.txt
COPY . .

CMD uvicorn deploy_inference:app --host 0.0.0.0 --port 8080