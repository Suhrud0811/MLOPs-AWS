import json
from onnxInference import ColaONNXPredictor


inferencing_instance = ColaONNXPredictor("./models/model.onnx")


def lambda_handler(event, context):
    if "resource" in event.keys():
        body = event["body"]
        body = json.loads(body)
        print(f"Got the input: {body['sentence']}")
        response = inferencing_instance.predict(body["sentence"])
        return {
            "statusCode": 200,
            "headers": {},
            "body": json.dumps(response)
        }
    else:
        return inferencing_instance.predict(event["sentence"])
    