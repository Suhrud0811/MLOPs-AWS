FROM amazon/aws-lambda-python:3.8


ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY
ARG MODEL_DIR=./models
RUN mkdir $MODEL_DIR

ENV TRANSFORMERS_CACHE=$MODEL_DIR \
    TRANSFORMERS_VERBOSITY=error

ENV AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY

RUN yum install git -y && yum -y install gcc-c++
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt --no-cache-dir

COPY ./ ./
ENV PYTHONPATH "${PYTHONPATH}:./"
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
RUN pip install "dvc[s3]"


# configuring remote server in dvc

RUN dvc init --no-scm
RUN dvc remote add -d model-store s3://models-dvc/trained_models/


# pulling the trained model
RUN dvc pull dvcfiles/trained_model.dvc


RUN python lambda_handler.py
RUN chmod -R 0755 $MODEL_DIR
CMD [ "lambda_handler.lambda_handler"]



# COPY ./ /app
# WORKDIR /app

# # For AWS
# ARG AWS_ACCESS_KEY_ID
# ARG AWS_SECRET_ACCESS_KEY
# ARG AWS_DEFAULT_REGION



# # aws credentials configuration
# ENV AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
#     AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
#     AWS_DEFAULT_REGION=$AWS_DEFAULT_REGION

# RUN echo "AWS_ACCESS_KEY_ID: $AWS_ACCESS_KEY_ID" \
#     && echo "AWS_SECRET_ACCESS_KEY: $AWS_SECRET_ACCESS_KEY" \
#     && echo "REGION: $AWS_DEFAULT_REGION"


# # Install requirements
# # RUN pip install "dvc[gdrive]"
#  # since s3 is the remote storage
# RUN pip install "dvc[s3]"  
# RUN pip install -r requirements.txt

# # Initialize DVC
# RUN dvc init --no-scm -f

# # Configure S3 as remote storage
# RUN dvc remote add -d model-store s3://models-dvc/trained_models/

# # Configure Google Drive as remote storage
# # RUN dvc remote add -d storage gdrive://1cMM3EbI0cl37pUdOFMUrZTEpxCxr5VPV
# # RUN dvc remote modify storage gdrive_use_service_account true
# RUN dvc remote modify model-store region us-east-2  # Replace with your region


# RUN cat .dvc/config


# # pulling the trained model
# RUN dvc pull ./models/model.onnx.dvc --verbose


# # FOR PASSING CREDS TO GOOGLE DRIVE
# # Pass the creds.json file to the container during build
# # ARG TYPE
# # ARG PROJECT_ID
# # ARG PRIVATE_KEY_ID
# # ARG PRIVATE_KEY
# # ARG CLIENT_EMAIL
# # ARG CLIENT_ID
# # ARG AUTH_URI
# # ARG TOKEN_URI
# # ARG AUTH_PROVIDER_X509_CERT_URL
# # ARG CLIENT_X509_CERT_URL
# # ARG UNIVERSE_DOMAIN



# # # Create the creds.json file (assuming you already have this part)
# # RUN printf '{\n'\
# # '  "type": "%s",\n'\
# # '  "project_id": "%s",\n'\
# # '  "private_key": "%s",\n'\
# # '  "private_key_id": "%s",\n'\
# # '  "client_email": "%s",\n'\
# # '  "client_id": "%s",\n'\
# # '  "auth_uri": "%s",\n'\
# # '  "token_uri": "%s",\n'\
# # '  "auth_provider_x509_cert_url": "%s",\n'\
# # '  "client_x509_cert_url": "%s",\n'\
# # '  "universe_domain": "%s"\n'\
# # '}' \
# # "${TYPE}" \
# # "${PROJECT_ID}" \
# # "${PRIVATE_KEY}" \
# # "${PRIVATE_KEY_ID}" \
# # "${CLIENT_EMAIL}" \
# # "${CLIENT_ID}" \
# # "${AUTH_URI}" \
# # "${TOKEN_URI}" \
# # "${AUTH_PROVIDER_X509_CERT_URL}" \
# # "${CLIENT_X509_CERT_URL}" \
# # "${UNIVERSE_DOMAIN}" > /app/creds.json

# # RUN cat /app/creds.json


# # # Configure DVC to use the credentials
# # RUN dvc remote modify storage gdrive_service_account_json_file_path /app/creds.json
# # # Pull the trained model
# # RUN dvc pull models/trained-model.ckpt.dvc

# # Expose the necessary port
# EXPOSE 8000

# # Command to start the application
# CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
