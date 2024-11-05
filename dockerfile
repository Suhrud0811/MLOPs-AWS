# Stage 1: Build with amazonlinux to install required packages
FROM amazonlinux:2 AS build

# Install git, gcc-c++, Python, and pip
RUN yum install -y git gcc-c++ && \
    amazon-linux-extras enable python3.8 && \
    yum install -y python3.8 && \
    python3.8 -m ensurepip && \
    yum clean all

# Set Python and Pip aliases to use python3 and pip3 as commands
RUN ln -s /usr/bin/python3.8 /usr/bin/python && \
    ln -s /usr/bin/python3.8 /usr/bin/python3 && \
    ln -s /usr/local/bin/pip3 /usr/bin/pip3

# Copy in requirements and install dependencies to /build
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt --no-cache-dir -t /build

# Install DVC with S3 support
RUN pip3 install "dvc[s3]" --no-cache-dir -t /build

# Stage 2: Use the Lambda-compatible base image and copy dependencies
FROM amazon/aws-lambda-python:3.8

# Set up argument and environment variables
ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY
ARG MODEL_DIR=./models

ENV TRANSFORMERS_CACHE=$MODEL_DIR \
    TRANSFORMERS_VERBOSITY=error \
    AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
    AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
    PYTHONPATH="${PYTHONPATH}:./" \
    LC_ALL=C.UTF-8 \
    LANG=C.UTF-8

# Copy dependencies from the build stage
COPY --from=build /build /opt/python

# Create and set permissions for the model directory
RUN mkdir -p $MODEL_DIR && chmod -R 0755 $MODEL_DIR

# Initialize DVC and configure remote server
RUN dvc init --no-scm
RUN dvc remote add -d model-store s3://models-dvc/trained_models/

# Copy the application code
COPY ./ ./

# Pull the trained model
RUN dvc pull dvcfiles/trained_model.dvc

# Run the application
CMD [ "lambda_handler.lambda_handler"]
