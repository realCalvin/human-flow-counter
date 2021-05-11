import boto3
import base64

def detect_face(filename):
    client = boto3.client('rekognition', region_name='us-west-2')

    with open (filename, 'rb') as image:
        encoded = base64.b64encode(image.read())
        binary = base64.decodebytes(encoded)

    response = client.detect_faces(Image={'Bytes': binary}, Attributes=["ALL"])
    try:
        print(response['FaceDetails'][0]['Gender'])
    except:
        print('No face detected')