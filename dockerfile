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
ARG TYPE
ARG PROJECT_ID
ARG PRIVATE_KEY_ID
ARG PRIVATE_KEY
ARG CLIENT_EMAIL
ARG CLIENT_ID
ARG AUTH_URI
ARG TOKEN_URI
ARG AUTH_PROVIDER_X509_CERT_URL
ARG CLIENT_X509_CERT_URL
ARG UNIVERSE_DOMAIN



# Set environment variables to match Google service account JSON fields
RUN echo '{
  "type": "'${GDRIVE_TYPE}'",
  "project_id": "'${GDRIVE_PROJECT_ID}'",
  "private_key": "'${GDRIVE_PRIVATE_KEY}'",
  "client_email": "'${GDRIVE_CLIENT_EMAIL}'",
  "client_id": "'${GDRIVE_CLIENT_ID}'",
  "auth_uri": "'${GDRIVE_AUTH_URI}'",
  "token_uri": "'${GDRIVE_TOKEN_URI}'",
  "auth_provider_x509_cert_url": "'${GDRIVE_AUTH_PROVIDER_CERT_URL}'",
  "client_x509_cert_url": "'${GDRIVE_CLIENT_CERT_URL}'"
}' > /app/creds.json

# Set the path for the service account json
RUN dvc remote modify storage gdrive_service_account_json_file_path /app/creds.json
# Pull the trained model
RUN dvc pull models/trained-model.ckpt.dvc

# Expose the necessary port
EXPOSE 8000

# Command to start the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
