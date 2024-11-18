import json
import os
from datetime import datetime
import cv2


# File path
file_path = "data.json"

# Default empty data (JSON structure)
default_data = []

# Function to handle attendance
def attendance(unique_names, attend_names,frame_resized):
    current_datetime = datetime.now()
    

    # Print the current date and time
    print("Current Date and Time:", current_datetime)

    # Load the existing data or create a new file if not exists
    if not os.path.exists(file_path):
        with open(file_path, "w") as json_file:
            json.dump(default_data, json_file, indent=4)
        print("New JSON file created with empty data.")
    else:
        print("File already exists.")

    # Step 2: Load data from the JSON file
    with open(file_path, "r") as json_file:
        data = json.load(json_file)

    # Create a set of existing names for faster lookup
    existing_names = {entry["name"] for entry in data}

    # Step 3: Process each unique name
    for person in unique_names:
        print("The detected person was:", person)
        if person != "Unknown" and person not in existing_names:
            # Add new person data if not already in the list
            new_entry = {
                "name": person,
                "date_time": current_datetime.strftime("%Y-%m-%d %H:%M:%S")
            }
            data.append(new_entry)
            existing_names.add(person)  # Add the person to the set of existing names
            
            # Save the updated data back to the file
            with open(file_path, "w") as json_file:
                json.dump(data, json_file, indent=4)
            cv2.imwrite(f"frames/{person}.png",frame_resized)
            
            print("Data added/updated successfully.")
    
 

    print("Updated attendance:", attend_names)