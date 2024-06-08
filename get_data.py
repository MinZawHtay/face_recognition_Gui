import psycopg2
import cv2
import face_recognition
import os

known_face_encodings = []
known_face_names = []

def connectDB():
    try:
        connection = psycopg2.connect(
            host="localhost",
            port=5432,
            database="face_recognition",
            user="local_admin",
            password="admin@123"
        )
        return connection
    except Exception as e:
        print("Error connecting to the database:", e)
        return None

conn = connectDB()

if conn:
    try:
        cur = conn.cursor()

        # Execute the queries
        cur.execute("SELECT name, image_path FROM user_inputs;")
        rows = cur.fetchall()

        # Process each row
        for row in rows:
            name, image_path = row
            print("Name:", name)
            print("Image Path:", image_path)
            
            if os.path.exists(image_path):
                image = face_recognition.load_image_file(image_path)
                encodings = face_recognition.face_encodings(image)
                if encodings:
                    known_face_encodings.append(encodings[0])
                    known_face_names.append(name)
                else:
                    print(f"No face found in the image {image_path}")
            else:
                print(f"Image path {image_path} does not exist.")

        cur.close()
    except Exception as e:
        print("Error executing the queries:", e)
    finally:
        conn.close()
else:
    print("Failed to connect to the database.")

cap = cv2.VideoCapture(0)  # 0 represents the default camera (usually the built-in webcam)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Find all face locations and encodings in the frame
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    for i, (top, right, bottom, left) in enumerate(face_locations):
        # Compare the face encoding to known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encodings[i])

        name = "Unknown"
        confidence = 0  # Initialize confidence to 0

        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index].upper()  # Convert name to uppercase

            # Calculate the face distance (lower value indicates a better match)
            face_distances = face_recognition.face_distance(known_face_encodings, face_encodings[i])
            confidence = 100 - (face_distances[first_match_index] * 100)

        # Draw a rectangle and label on the image with green background and bold green text
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(frame, (left, top - 20), (right, top), (0, 255, 0), cv2.FILLED)  # Green background
        text = f"{name} ({confidence:.2f}%)"
        cv2.putText(frame, text, (left + 6, top - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
