from ultralytics import YOLO
from datetime import datetime
import cv2
import numpy as np
import easyocr
from googleapiclient.discovery import build
import pickle
import os.path
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from sort.sort import Sort


# Initialization for Google Sheets API
def init_google_sheets_api():
    creds = None
    token_path = 'F:\\yo2\\anpr\\token.pickle'
    credentials_path = 'auth2.Youre path.json'

    if os.path.exists(token_path):
        with open(token_path, 'rb') as token:
            creds = pickle.load(token)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(credentials_path, ['https://www.googleapis.com/auth/spreadsheets'])
            creds = flow.run_local_server(port=0)

        with open(token_path, 'wb') as token:
            pickle.dump(creds, token)

    service = build('sheets', 'v4', credentials=creds)
    return service


# Function to append data to Google Sheets
def append_to_sheet(service, spreadsheet_id, range_name, values):
    body = {
        'values': values
    }
    result = service.spreadsheets().values().append(
        spreadsheetId=spreadsheet_id, range=range_name,
        valueInputOption="RAW", body=body).execute()
    print('{0} cells appended.'.format(result.get('updates').get('updatedCells')))


results = {}
service = init_google_sheets_api()
tracked_vehicles = {}
# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# Load models
coco_model = YOLO('yolov8n.pt')
license_plate_detector = YOLO("license_plate_detector.pt")
tracker = Sort()

# Access camera
cap = cv2.VideoCapture(1)

vehicles = [2, 3, 5, 7]
frame_nmr = 0


while True:
    ret, frame = cap.read()
    if ret:
        current_results = {}  # Use this to store results for the current frame only

        # Detect vehicles
        detections = coco_model(frame)[0]
        bboxes = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                bboxes.append([x1, y1, x2, y2, score])

        # Update the tracker based on detections
        if bboxes:
            trackers = tracker.update(np.array(bboxes))
            for track in trackers:
                x1, y1, x2, y2, track_id = map(int, track)
                tracked_vehicles[track_id] = (x1, y1, x2, y2)  # Store the bounding box for each tracked vehicle
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)  # Draw blue bounding box for vehicles
                cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Detect license plates
        license_plates = license_plate_detector(frame)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            # Determine which tracked vehicle (if any) this license plate belongs to
            current_track_id = None
            for track_id, bbox in tracked_vehicles.items():
                t_x1, t_y1, t_x2, t_y2 = bbox
                if t_x1 <= x1 <= t_x2 and t_y1 <= y1 <= t_y2:  # If license plate is within the tracked vehicle's bounding box
                    current_track_id = track_id
                    break

            # Highlight license plate on the main video frame
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 255), 4)  # Drawing a rectangle

            # Crop license plate for OCR
            license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]

            try:
                # Read license plate number using EasyOCR
                detections = reader.readtext(license_plate_crop)
                for detection in detections:
                    license_plate_text = detection[1]
                    conf = detection[2]

                    if conf >= 0.8:
                        print(f"License Plate: {license_plate_text}, Confidence: {conf}")
                        current_results['license_plate'] = {
                            'text': license_plate_text,
                            'bbox': [x1, y1, x2, y2],
                            'conf': conf,
                            'track_id': current_track_id  # Store the tracking ID
                        }

                        # Annotate the frame with license plate text:
                        cv2.putText(frame, license_plate_text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                                    (0, 255, 255), 3)
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255),
                                      3)  # yellow rectangle

            except Exception as e:
                print(f"EasyOCR Error: {e}")

        if 'license_plate' in current_results:
            license_plate_data = current_results['license_plate']
            plate_text = license_plate_data['text']
            conf = license_plate_data['conf']
            x_center = round((x1 + x2) / 2)
            y_center = round((y1 + y2) / 2)
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # Add this to get the current timestamp
            append_to_sheet(service, 'Youre sheet Link',
                            [[plate_text, conf, timestamp, x_center, y_center, current_track_id]])  # Append tracking ID
        # Resize the frame before displaying
        height, width = frame.shape[:2]
        new_width = int(width * 0.5)  # 50% of the original width
        new_height = int(height * 0.5)  # 50% of the original height
        resized_frame = cv2.resize(frame, (new_width, new_height))

        # Display the resized frame
        cv2.imshow('Resized Live Feed', resized_frame)

        frame_nmr += 1

    # Exit when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
