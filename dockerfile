FROM huggingface/transformers-pytorch-cpu:latest

COPY ./ /app
WORKDIR /app

# Install requirements
RUN pip install "dvc[gdrive]"
RUN pip install -r requirements.txt

# Initialize DVC
RUN dvc init --no-scm -f
RUN dvc remote add -d storage gdrive://1cMM3EbI0cl37pUdOFMUrZTEpxCxr5VPV
RUN dvc remote modify storage gdrive_use_service_account true

# Pass the creds.json file to the container during build
ARG GDRIVE_CREDS
RUN echo "${GDRIVE_CREDS}" | base64 --decode > creds.json

# Set the path for the service account json
RUN dvc remote modify storage gdrive_service_account_json_file_path creds.json

# Pull the trained model
RUN dvc pull models/trained-model.ckpt.dvc

# Expose the necessary port
EXPOSE 8000

# Command to start the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
