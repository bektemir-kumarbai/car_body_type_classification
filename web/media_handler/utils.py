import os
import secrets
from typing import Tuple
import aiofiles

import base64
from io import BytesIO

from web.ai_service.body_classification import predict

RANDOM_STRING_CHARS = "1234567890QWERTYUIOPASDFGHJKLZXCVBNM"


def get_random_string(length, allowed_chars=RANDOM_STRING_CHARS) -> str:
    """
    Description: Generates a random string of a specified length using the provided characters.
    params:
        length: An integer specifying the length of the random string to generate.
        allowed_chars: A string containing characters that may be used in the random string. Defaults to a predefined string of uppercase letters and numbers.
    return: A string representing the randomly generated string.
    """
    return "".join(secrets.choice(allowed_chars) for i in range(length))

async def save_file_websocket_connection(file):
    unique_code = get_random_string(length=6)
    filepath = f"web/media/{unique_code}/{file}"

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    url: str = str(filepath.replace("web/", ""))

    async with aiofiles.open(filepath, "wb") as buffer:
        await buffer.write(await file.read())

    return url, filepath


async def save_binary(file_base64: str, filename: str) -> Tuple[str, str]:
    """
    Description: Asynchronously saves a base64-encoded image to a directory, generating a unique filename.
    params:
       file_base64: The base64-encoded content of the image to be saved.
       filename: The original filename of the image.
    return: A tuple containing the URL and filepath of the saved file. The URL is a relative path excluding the "web/" prefix.
    """
    unique_code = get_random_string(length=6)
    filepath = f"web/media/{unique_code}_{filename}"

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    url: str = str(filepath.replace("web/", ""))

    image_data = base64.b64decode(file_base64)
    buffer = BytesIO(image_data)

    async with aiofiles.open(filepath, "wb") as file:
        await file.write(buffer.getvalue())

    return url, filepath

async def save_file(file) -> Tuple[str, str]:
    """
    Description: Asynchronously saves a file to a directory, generating a unique filename.
    params:
        file: The file to be saved.
    return: A tuple containing the URL and filepath of the saved file. The URL is a relative path excluding the "web/" prefix.
    """
    unique_code = get_random_string(length=6)
    filepath = f"web/media/{unique_code}_{file.filename}"
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    url: str = str(filepath.replace("web/", ""))

    async with aiofiles.open(filepath, "wb") as buffer:
        await buffer.write(await file.read())

    return url, filepath


async def process_image(img_path: str) -> dict:
    """
    Description: Processes an image to detect a car and its attributes, as well as the license plate.
    params:
        img_path: A string representing the path to the image file.
    return: A dictionary with data if a car is not detected, or a tuple containing the license plate information and car attributes.
    """
    predictions = predict(img_path)
    return predictions


