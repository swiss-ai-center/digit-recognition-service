from common_code.config import get_settings
from common_code.logger.logger import get_logger, Logger
from common_code.service.models import Service
from common_code.service.enums import ServiceStatus
from common_code.common.enums import FieldDescriptionType, ExecutionUnitTagName, ExecutionUnitTagAcronym
from common_code.common.models import FieldDescription, ExecutionUnitTag
from common_code.tasks.models import TaskData
# Imports required by the service's model
from keras import models
import cv2
import numpy as np

api_description = """
Recognizes a digit in an image using mnist trained model.
"""
api_summary = """
Recognizes a digit in an image using mnist trained model.
"""

api_title = "Digit Recognition API."
version = "1.0.0"

settings = get_settings()


class MyService(Service):
    """
    Digit recognition service model
    """

    # Any additional fields must be excluded for Pydantic to work
    _model: object
    _logger: Logger

    def __init__(self):
        super().__init__(
            name="Digit Recognition",
            slug="digit-recognition",
            url=settings.service_url,
            summary=api_summary,
            description=api_description,
            status=ServiceStatus.AVAILABLE,
            data_in_fields=[
                FieldDescription(name="image", type=[FieldDescriptionType.IMAGE_PNG, FieldDescriptionType.IMAGE_JPEG]),
            ],
            data_out_fields=[
                FieldDescription(name="digit", type=[FieldDescriptionType.TEXT_PLAIN]),
            ],
            tags=[
                ExecutionUnitTag(
                    name=ExecutionUnitTagName.IMAGE_RECOGNITION,
                    acronym=ExecutionUnitTagAcronym.IMAGE_RECOGNITION
                ),
            ],
            has_ai=True,
            docs_url="https://docs.swiss-ai-center.ch/reference/services/digit-recognition/",
        )
        self._logger = get_logger(settings)
        self._model = models.load_model(os.path.join(os.path.dirname(__file__), "..", "mnist_model.h5"))

    def process(self, data):
        # Get raw image data
        raw = data["image"].data
        # Convert to image object
        image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
        # Get the shape of the model
        shape = self._model.input_shape[1]
        # Compute the size
        size = int(np.sqrt(shape))
        # Resize the image to the size of the model
        resized = cv2.resize(image, (size, size))
        # Normalize the image
        normalized = resized / 255.0
        # Reshape the image
        flattened = np.reshape(normalized, [-1, shape])
        # Predict the image
        prediction = self._model.predict(flattened)
        # Get the index of the highest probability
        guessed = np.argmax(prediction)

        # Save the results in a type that can be encoded later (str, json, ...)
        return {
            "digit": TaskData(
                data=str(guessed),
                type=FieldDescriptionType.TEXT_PLAIN
            )
        }
