FROM python:3.9
WORKDIR /src
COPY . .
RUN pip install -r requirements.txt --no-deps
ENTRYPOINT [ "streamlit","run","Home.py" ]