import cv2
import numpy as np
import math
##########################
#NB: switch env name to april_gen NOT real3D
##########################
# Parameters
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)  # AprilTag dictionary
tag_size_mm = 12  # Tag size in millimeters 14.4
tags_per_row = 11  # Number of tags in one row 11
tags_per_column = 16 # Number of tags in one column 16
margin_mm = 6  # Margin between tags in millimeters 4
dpi = 300  # Resolution for printing (dots per inch)
page_width_mm = 210  # Width of an A4 page in millimeters
page_height_mm = 297  # Height of an A4 page in millimeters
start_id = 0  # Starting tag ID
end_id = 586  # Last tag ID (total tags = end_id - start_id + 1)

# Conversions
mm_to_pixels = dpi / 25.4
tag_size_pixels = int(tag_size_mm * mm_to_pixels)
margin_pixels = int(margin_mm * mm_to_pixels)
page_width_pixels = int(page_width_mm * mm_to_pixels)
page_height_pixels = int(page_height_mm * mm_to_pixels)

# Calculate grid spacing
grid_width = tags_per_row * (tag_size_pixels + margin_pixels) - margin_pixels
grid_height = tags_per_column * (tag_size_pixels + margin_pixels) - margin_pixels

# Ensure the grid fits within the page dimensions
if grid_width > page_width_pixels or grid_height > page_height_pixels:
    raise ValueError("The grid size exceeds the page dimensions. Reduce tags per row/column or margins.")

# Create pages of tags
current_id = start_id
tags_per_page = tags_per_row * tags_per_column
page_number = 1

while current_id <= end_id:
    # Create a blank page
    page = np.ones((page_height_pixels, page_width_pixels), dtype=np.uint8) * 255  # White background

    # Draw tags on the grid
    for row in range(tags_per_column):
        
        for col in range(tags_per_row):
            if current_id > end_id:
                break

            # Calculate tag position
            x = col * (tag_size_pixels + margin_pixels) + margin_pixels//2
            y = row * (tag_size_pixels + margin_pixels) + margin_pixels//2

            # Generate the tag
            tag_img = np.zeros((tag_size_pixels, tag_size_pixels), dtype=np.uint8)
            cv2.aruco.generateImageMarker(aruco_dict, current_id, tag_size_pixels, tag_img)

            # Place the tag on the page
            page[y:y + tag_size_pixels, x:x + tag_size_pixels] = tag_img
            page[y + tag_size_pixels + margin_pixels//2, x - margin_pixels//2:x + tag_size_pixels + margin_pixels//2] = 0
            page[y - margin_pixels//2:y + tag_size_pixels + margin_pixels//2, x + tag_size_pixels + margin_pixels//2] = 0
            current_id += 1

    # Save the page as an image
    output_filename = f"apriltags_page_test_{page_number}.png"
    cv2.imwrite(output_filename, page)
    print(f"Saved: {output_filename}")
    page_number += 1

