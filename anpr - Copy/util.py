import csv
import os
import string
import easyocr
from datetime import datetime

# Initialize the OCR reader
reader = easyocr.Reader(['en'], gpu=False)

# Mapping dictionaries for character conversion
dict_char_to_int = {'O': '0',
                    'I': '1',
                    'J': '3',
                    'A': '4',
                    'G': '6',
                    'S': '5'}

dict_int_to_char = {'0': 'O',
                    '1': 'I',
                    '3': 'J',
                    '4': 'A',
                    '6': 'G',
                    '5': 'S'}


import csv
import os

def write_csv(data, filename):
    """
    Write results to a CSV file
    :param data: Data to be written. Expected format: {frame_number: {'license_plate': {'text': 'text_value', 'bbox': [x1, y1, x2, y2], 'confidence': confidence_value}}}
    :param filename: Name of the CSV file
    """
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]  # format the current time with milliseconds

    if not os.path.isfile(filename):
        # If the CSV doesn't exist, create one and write headers
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Timestamp", "Frame Number", "License Plate Text", "Bounding Box", "Confidence"])

    # Append data to CSV
    with open(filename, 'a', newline='') as file:
        writer = csv.writer(file)
        for frame, details in data.items():
            lp = details.get('license_plate', {})
            text = lp.get('text', 'N/A')
            bbox = lp.get('bbox', [])
            conf = lp.get('conf', 'N/A')  # Extract confidence value
            writer.writerow([current_time, frame, text, bbox, conf])
        file.flush()  # Flush the file buffer to ensure the data is written to the disk

    print(f"Data with timestamp written to {filename}")


    print(f"Data with timestamp written to {filename}")



def license_complies_format(text):
    """
    Check if the license plate text complies with the required format.

    Args:
        text (str): License plate text.

    Returns:
        bool: True if the license plate complies with the format, False otherwise.
    """
    # Assuming a license plate is valid if it contains between 5 to 10 alphanumeric characters.
    if 5 <= len(text) <= 10 and text.isalnum():
        return True
    else:
        return False

def format_license(text):
    """
    Format the license plate text by converting characters using the mapping dictionaries.

    Args:
        text (str): License plate text.

    Returns:
        str: Formatted license plate text.
    """
    license_plate_ = ''
    mapping = {0: dict_int_to_char, 1: dict_int_to_char, 4: dict_int_to_char, 5: dict_int_to_char, 6: dict_int_to_char,
               2: dict_char_to_int, 3: dict_char_to_int}
    for j in [0, 1, 2, 3, 4, 5, 6]:
        if text[j] in mapping[j].keys():
            license_plate_ += mapping[j][text[j]]
        else:
            license_plate_ += text[j]

    return license_plate_


def read_license_plate(license_plate_crop):
    """
    Read the license plate text from the given cropped image.

    Args:
        license_plate_crop (PIL.Image.Image): Cropped image containing the license plate.

    Returns:
        tuple: Tuple containing the formatted license plate text and its confidence score.
    """

    detections = reader.readtext(license_plate_crop)

    for detection in detections:
        bbox, text, score = detection

        text = text.upper().replace(' ', '')

        if license_complies_format(text):
            return format_license(text), score

    return None, None


def get_car(license_plate, vehicle_track_ids):
    """
    Retrieve the vehicle coordinates and ID based on the license plate coordinates.

    Args:
        license_plate (tuple): Tuple containing the coordinates of the license plate (x1, y1, x2, y2, score, class_id).
        vehicle_track_ids (list): List of vehicle track IDs and their corresponding coordinates.

    Returns:
        tuple: Tuple containing the vehicle coordinates (x1, y1, x2, y2) and ID.
    """
    x1, y1, x2, y2, score, class_id = license_plate

    foundIt = False
    for j in range(len(vehicle_track_ids)):
        xcar1, ycar1, xcar2, ycar2, car_id = vehicle_track_ids[j]

        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            car_indx = j
            foundIt = True
            break

    if foundIt:
        return vehicle_track_ids[car_indx]

    return -1, -1, -1, -1, -1
