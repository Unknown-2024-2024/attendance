# import cv2
# from simple_facerec import SimpleFacerec
# from attendance import attendance
# from prefect import flow, task

# @task
# def process_frame_task(frame, attend_names):
#     """Process a single frame to detect faces and mark attendance."""
#     # Create SimpleFacerec instance within the task
#     sfr = SimpleFacerec()
#     sfr.load_encoding_images("images/")

#     # Detect faces
#     face_locations, face_names = sfr.detect_known_faces(frame)

#     for face_loc, name in zip(face_locations, face_names):
#         y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]
#         cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
#         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)

#     frame_resized = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

#     if face_names:
#         unique_names = list(set(face_names))
#         attendance(unique_names, attend_names, frame_resized)

#     return frame_resized


# @flow(log_prints=True)
# def main_flow():
#     # Load Camera
#     cap = cv2.VideoCapture("rtsp://suprisridhar@gmail.com:AshSri2123@192.168.1.75/stream1", cv2.CAP_FFMPEG)

#     if not cap.isOpened():
#         print("Error: Could not open video source")
#         return

#     attend_names = {}

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Failed to grab frame")
#             break  # Handle the error appropriately

#         # Call the task to process the frame
#         frame_resized_future = process_frame_task.submit(frame, attend_names)
#         frame_resized = frame_resized_future.result()  # Wait for the task to complete

#         cv2.imshow("Frame", frame_resized)

#         key = cv2.waitKey(1)
#         if key == 27:  # Escape key to exit
#             break

#     cap.release()
#     cv2.destroyAllWindows()


# if __name__ == "__main__":
#     main_flow()


import cv2
from simple_facerec import SimpleFacerec
from attendance import attendance
from prefect import flow


@flow(log_prints=True)
def main():

    # Encode faces from a folder
    sfr = SimpleFacerec()
    sfr.load_encoding_images("images/")

    # Load Camera
    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture("rtsp://suprisridhar@gmail.com:AshSri2123@192.168.1.75/stream1", cv2.CAP_FFMPEG)
    # cap.set(cv2.CAP_PROP_BUFFERSIZE, 100) 

    if not cap.isOpened():
        
        print("Error: Could not open video source")


    attend_names={}


    # Set the desired frame width and height

    while True:
        
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break  # or handle the error appropriately

        # Detect Faces
        # cv2.resize(frame,(640,480))
        face_locations, face_names = sfr.detect_known_faces(frame)
        
            
        for face_loc, name in zip(face_locations, face_names):
            y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

            cv2.putText(frame, name,(x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)
        frame_resized = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        if face_names:
            unique_names = list(set(face_names))
            attendance(unique_names,attend_names,frame_resized)
        cv2.imshow("Frame", frame_resized)

        key = cv2.waitKey(1)
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows() 
if __name__ == "__main__":
    main()
