FROM python:3.11-slim

EXPOSE 5000/tcp

WORKDIR /LLM_PHISHING_DETECTION/

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . .

RUN export PYTHONPATH=${PYTHONPATH}:LLM_phishing_detection

CMD ["flask", "run", "--host", "0.0.0.0" ]