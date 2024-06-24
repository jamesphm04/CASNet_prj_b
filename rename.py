import os
import shutil

def rename_and_copy_images(source_folder, destination_folder):
    """
    Rename images in a folder based on their class and copy them to a new folder.

    Parameters:
    source_folder (str): Path to the folder containing images.
    destination_folder (str): Path to the folder where renamed images will be copied.
    """
    # Create the destination folder if it doesn't exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Get a list of all files in the source folder
    files = os.listdir(source_folder)
    # Filter only image files
    image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

    # Dictionary to store counts for each class
    class_counts = {'0': 0, '1': 0}

    # Rename images and create label.txt file
    with open(os.path.join(os.path.dirname(destination_folder), 'label.txt'), 'w') as label_file:
        for image_file in image_files:
            # Split filename into parts
            parts = image_file.split('_')
            # Extract class from filename
            image_class = parts[-1].split('.')[0]

            # Increment count for the class
            class_counts[image_class] += 1

            # Copy the image to the destination folder and rename it
            new_name_prefix = 'deform' if image_class == '1' else 'truth'
            new_name = f"{new_name_prefix}_{class_counts[image_class]}.png"
            old_path = os.path.join(source_folder, image_file)
            new_path = os.path.join(destination_folder, new_name)
            shutil.copy(old_path, new_path)

            # Write label information to label.txt
            label_file.write(f"SC2PC/{new_name} {'1' if new_name_prefix == 'deform' else '0'}\n")



# Source folder containing the original images
source_folder = "data/converted/SC2PC"
# Destination folder where renamed images will be copied
destination_folder = "data/SC_gen_data/SC2PC"

rename_and_copy_images(source_folder, destination_folder)