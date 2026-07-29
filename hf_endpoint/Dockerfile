FROM python:3.11-slim

RUN useradd --create-home --uid 1000 appuser
USER appuser
ENV HOME=/home/appuser \
    PATH=/home/appuser/.local/bin:$PATH \
    HF_HOME=/home/appuser/.cache/huggingface
WORKDIR $HOME/app

COPY --chown=appuser:appuser requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt
COPY --chown=appuser:appuser app.py .

EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
