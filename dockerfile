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
ENV type=$TYPE
ENV project_id=$PROJECT_ID
ENV private_key_id=$PRIVATE_KEY_ID
ENV private_key=$PRIVATE_KEY
ENV client_email=$CLIENT_EMAIL
ENV client_id=$CLIENT_ID
ENV auth_uri=$AUTH_URI
ENV token_uri=$TOKEN_URI
ENV auth_provider_x509_cert_url=$AUTH_PROVIDER_X509_CERT_URL
ENV client_x509_cert_url=$CLIENT_X509_CERT_URL
ENV universe_domain=$GDRIVE_UNIVERSE_DOMAIN

# Set the path for the service account json
RUN dvc remote modify storage gdrive_use_service_account true

# Pull the trained model
RUN dvc pull models/trained-model.ckpt.dvc

# Expose the necessary port
EXPOSE 8000

# Command to start the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
