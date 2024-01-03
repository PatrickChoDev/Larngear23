FROM python:3.12-bullseye
WORKDIR /app
COPY fonts/ ./fonts/
COPY POSTER/ ./POSTER/
COPY .chainlit/ ./.chainlit/
COPY chainlit.md ./
COPY environment.txt ./
RUN pip install -r environment.txt
COPY app.py ./

EXPOSE 8000
CMD ["chainlit","run","app.py"]