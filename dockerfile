FROM huggingface/transformers-pytorch-cpu:latest

COPY ./ /app
WORKDIR /app

# Install requirements
RUN pip install "dvc[gdrive]"
RUN pip install -r requirements.txt

# Initialize DVC
RUN dvc init --no-scm -f
RUN dvc remote add -d storage gdrive://1cMM3EbI0cl37pUdOFMUrZTEpxCxr5VPV

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
EXPOSE 8000

# Create creds.json file using the secret passed as an environment variable
CMD /bin/bash -c 'echo "$GDRIVE_CREDS" > creds.json && \
    dvc remote modify storage gdrive_service_account_json_file_path creds.json && \
    dvc pull models/trained-model.ckpt.dvc && \
    uvicorn app:app --host 0.0.0.0 --port 8000'



