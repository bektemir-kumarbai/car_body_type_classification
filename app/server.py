import grpc
from concurrent import futures
import sys
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent.absolute())
sys.path.insert(0, project_root)
import body_classify_pb2
import body_classify_pb2_grpc
from PIL import Image
import io
from app.ai.predictor import CarBodyTypePredictor
from app.core.config import Settings
settings = Settings()
SECRET_TOKEN = settings.SECRET_TOKEN
predictor = CarBodyTypePredictor(settings.MODEL_PATH)

class ImageProcessingServicer(body_classify_pb2_grpc.ImageProcessingServiceServicer):
    def ProcessImage(self, request, context):
        if request.token != SECRET_TOKEN:
            context.set_code(grpc.StatusCode.UNAUTHENTICATED)
            context.set_details("Invalid authentication token")
            print("Invalid authentication token")
            return body_classify_pb2.ImageResponse()

        image = Image.open(io.BytesIO(request.image_data))

        try:
            prediction = predictor.predict(image)
            
            predicted_label = prediction['car_type_body']
            confidence_score = prediction['car_type_body_score']
            
        except Exception as ex:
            print(ex)
            predicted_label = None
            confidence_score = None
            
        return body_classify_pb2.ImageResponse(
            car_type_body=predicted_label,
            car_type_body_score=confidence_score
        )

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    body_classify_pb2_grpc.add_ImageProcessingServiceServicer_to_server(ImageProcessingServicer(), server)
    server.add_insecure_port("[::]:" + str(settings.APP_PORT))
    server.start()
    print("🔒 gRPC server started on port " + str(settings.APP_PORT))
    server.wait_for_termination()

if __name__ == "__main__":
    serve()