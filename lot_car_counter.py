from roboflow import Roboflow
import supervision as sv
import cv2
from PIL import Image
from datetime import datetime
import os 
import csv
import numpy as np
import json 
import math


def get_date_taken(image_path):
    try:
        # Open the image file
        with Image.open(image_path) as img:
            # Get the creation time from the metadata
            exif_data = img._getexif()
            if exif_data:
                # Check if the date and time information is available
                if 36867 in exif_data:
                    date_taken = exif_data[36867]
                    # Convert the date and time string to a datetime object
                    date_taken = datetime.strptime(date_taken, "%Y:%m:%d %H:%M:%S")
                    return date_taken
                else:
                    return "NO DATA"
            else:
                return "NO DATA"
    except Exception as e:
        return f"Error: {str(e)}"
    
def is_point_inside_polygon(x, y, poly):
    n = len(poly)
    inside = False

    p1x, p1y = poly[0]
    for i in range(1, n + 1):
        p2x, p2y = poly[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x) and p1y != p2y and p1x != p2x:
                    xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside

def euclidean_distance(point1, point2):

    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def export_to_csv(data, filename):
    # Extract header from the first dictionary in the list
    header = list(data[0].keys())

    # Open the CSV file for writing
    with open(filename, 'w', newline='') as csvfile:
        # Create a CSV writer object
        csv_writer = csv.DictWriter(csvfile, fieldnames=header)

        # Write the header to the CSV file
        csv_writer.writeheader()

        # Write each dictionary in the list to the CSV file
        for row in data:
            # Convert datetime objects to string
            row['datetime'] = row['datetime'].strftime('%Y-%m-%d %H:%M:%S')
            csv_writer.writerow(row)

def process_folder(folder_path):
    JSON_list = []
    dates_list = []
    for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')) and filename.lower()!="base.jpg": #make sure an image
                image_path = os.path.join(folder_path, filename) 
                print("Running detection on " , filename.lower())
                detections, datetime = detect_cars(image_path) #use NN to find number of cars 
                print("Detection complete. Appending data.")
                JSON_list.append(detections)
                dates_list.append(datetime)
    return JSON_list, dates_list

def detect_cars(image_path):
    print("Registering image", image_path)
    register(image_path) #creates registered.jpg
    print("Predicting from inference with ", image_path)
    result = model.predict("registered.jpg", confidence=10, overlap=50).json()
    detections = sv.Detections.from_inference(result)
    print("Extracting date and time data")
    datetimetaken = get_date_taken(image_path)
    return result, datetimetaken

def filter_areas(detections, times):
    mask_path = os.path.join(folder_path, "regions.json")
    # Load regions data
    with open(mask_path, 'r') as regions_file:
        regions_data = json.load(regions_file)

    # Initialize a dictionary to store parking counts
    counts_list = []

    # Loop through each detection in the list
    for detections_data, datetime in zip(detections, times):
        parking_counts = {parking_type: 0 for parking_type in regions_data}
        # Check if "predictions" key is present in the data
        if 'predictions' in detections_data:
            # Extract x and y coordinates for each detection
            #detection_coordinates = [(detection['x'], detection['y']) for detection in detections_data['predictions']]
            detection_coordinates = [
                (detection['x'] , (detection['y'] ))
                for detection in detections_data['predictions']
            ]
            print("filtering close overlaps for ", datetime)
            no_overlap_coordinates = filter_close_detections(detection_coordinates,30)
            # Check each coordinate against mask regions
            for parking_type, mask_coords in regions_data.items():
                for detection_coordinate in no_overlap_coordinates:
                    x, y = detection_coordinate
                    # Check if the center coordinate is within the mask region           
                    if is_point_inside_polygon(x,y, mask_coords):
                        # Increment the count for the corresponding parking type
                        parking_counts[parking_type] += 1
            print("Found ", parking_counts)   
            counts_dict = {'datetime': datetime, **parking_counts}
            counts_list.append(counts_dict)
            
    return counts_list

def filter_close_detections(detection_coordinates, distance_threshold):
    filtered_detections = []

    for i in range(len(detection_coordinates)):
        keep_detection = True

        for j in range(i + 1, len(detection_coordinates)):
            distance = euclidean_distance(detection_coordinates[i], detection_coordinates[j])

            if distance < distance_threshold:
                keep_detection = False
                break

        if keep_detection:
            filtered_detections.append(detection_coordinates[i])

    return filtered_detections

def register(input):
    # Convert to grayscale. 
    base_image_path = os.path.join(folder_path, "base.jpg")  
    img1_color = cv2.imread(input)  #
    img1 = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY) 
    img2 = cv2.cvtColor(cv2.imread(base_image_path), cv2.COLOR_BGR2GRAY) 
    height, width = img2.shape 

    # Create ORB detector with 5000 features. 
    orb_detector = cv2.ORB_create(5000) 

    # Find keypoints and descriptors. 
    # The first arg is the image, second arg is the mask 
    #  (which is not required in this case). 
    kp1, d1 = orb_detector.detectAndCompute(img1, None) 
    kp2, d2 = orb_detector.detectAndCompute(img2, None) 

    # Match features between the two images. 
    # We create a Brute Force matcher with  
    # Hamming distance as measurement mode. 
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True) 

    # Match the two sets of descriptors. 
    matches = matcher.match(d1, d2) 

    # Sort matches on the basis of their Hamming distance. 
    list(matches).sort(key = lambda x: x.distance) 

    # Take the top 90 % matches forward. 
    matches = matches[:int(len(matches)*0.9)] 
    no_of_matches = len(matches) 

    # Define empty matrices of shape no_of_matches * 2. 
    p1 = np.zeros((no_of_matches, 2)) 
    p2 = np.zeros((no_of_matches, 2)) 

    for i in range(len(matches)): 
        p1[i, :] = kp1[matches[i].queryIdx].pt 
        p2[i, :] = kp2[matches[i].trainIdx].pt 

    # Find the homography matrix. 
    homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC) 

    # Use this matrix to transform the 
    # colored image wrt the reference image. 
    transformed_img = cv2.warpPerspective(img1_color, 
                    homography, (width, height)) 
    
    cv2.imwrite("registered.jpg", transformed_img) 


try:
    rf = Roboflow(api_key="roWchbWIKRtDC9o7aeFR")
    project = rf.workspace().project("center_of_intersection")
    model = project.version(5).model
except:
    print("Unable to connect to Roboflow Project!")

folder_path = input("Enter the path to the image folder: ")

# Check if the provided path is valid
if os.path.exists(folder_path):
    base_image_path = os.path.join(folder_path, "base.jpg")  
    if os.path.exists(base_image_path):
        print("Found base image")
        mask_path = os.path.join(folder_path, "regions.json")
        if os.path.exists(mask_path):
            print("Found regions.json")
            print("Starting processing of directory")
            Detections_JSON_Data, times = process_folder(folder_path) 
            print("Detection complete! Starting region filtering")
            OccupancyByType_JSON_Data = filter_areas(Detections_JSON_Data, times)
            print("Writing data to CSV")
            export_to_csv(OccupancyByType_JSON_Data, "Report.csv")
        else:
            print("Mask JSON file not found...")
    else:
        print("Base image not found for registration...")
