from io import BytesIO
import base64
import numpy as np
from PIL import Image
import cv2


# converts image to base64 for input to torchserve api (used for b64 endpoint)
def transform_b64_to_image(b64):
    base64_decoded = base64.b64decode(b64)
    image = Image.open(BytesIO(base64_decoded))
    image_arr = np.array(image)
    return image_arr
