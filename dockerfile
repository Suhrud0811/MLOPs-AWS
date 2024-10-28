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



# Create the creds.json file (assuming you already have this part)
RUN printf '{\n'\
'  "type": "%s",\n'\
'  "project_id": "%s",\n'\
'  "private_key": "%s",\n'\
'  "private_key_id": "%s",\n'\
'  "client_email": "%s",\n'\
'  "client_id": "%s",\n'\
'  "auth_uri": "%s",\n'\
'  "token_uri": "%s",\n'\
'  "auth_provider_x509_cert_url": "%s",\n'\
'  "client_x509_cert_url": "%s",\n'\
'  "universe_domain": "%s"\n'\
'}' \
"${TYPE}" \
"${PROJECT_ID}" \
"${PRIVATE_KEY}" \
"${PRIVATE_KEY_ID}" \
"${CLIENT_EMAIL}" \
"${CLIENT_ID}" \
"${AUTH_URI}" \
"${TOKEN_URI}" \
"${AUTH_PROVIDER_X509_CERT_URL}" \
"${CLIENT_X509_CERT_URL}" \
"${UNIVERSE_DOMAIN}" > /app/creds.json



RUN cat /app/creds.json


# Configure DVC to use the credentials
RUN dvc remote modify storage gdrive_service_account_json_file_path /app/creds.json
# Pull the trained model
RUN dvc pull models/trained-model.ckpt.dvc

# Expose the necessary port
EXPOSE 8000

# Command to start the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
