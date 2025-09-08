FROM python:3.12

WORKDIR /app

RUN ["useradd", "-ms", "/bin/bash", "web_app"]
USER web_app

COPY requirements.txt .
RUN ["pip", "install", "--upgrade", "pip", "--no-warn-script-location"]
RUN ["pip", "install", "-r", "requirements.txt", "--no-warn-script-location"]

COPY . .

CMD ["python", "web_server.py"]
