from PIL import Image
import os

def resize_images(input_folder, new_size):
    """
    Resize all images in a folder and save the results to a new folder.

    Parameters:
    - input_folder (str): Path to the input folder containing image files.
    - new_size (tuple): Target size (width, height).
    """
    try:
        # Get the name of the input folder
        folder_name = os.path.basename(input_folder)

        # Create the output folder path based on the input folder name
        output_folder = os.path.join(os.path.dirname(input_folder), f"{folder_name}_resize_images")

        # Create the output folder if it doesn't exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Iterate over all files in the input folder
        for filename in os.listdir(input_folder):
            # Check if the file is an image (you may want to add more file format checks)
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                # Construct the full path for input and output images
                input_path = os.path.join(input_folder, filename)
                output_path = os.path.join(output_folder, filename)

                # Open the image file
                with Image.open(input_path) as img:
                    # Resize the image
                    resized_img = img.resize(new_size)

                    # Save the resized image to the output path
                    resized_img.save(output_path)

                    print(f"Image '{filename}' resized successfully and saved to {output_path}")

        return output_folder

    except Exception as e:
        print(f"Error: {e}")

# Example usage:
input_folder_path = r"F:\Dataset\TRI5004"
target_size = (256, 256)

output_folder_path = resize_images(input_folder_path, target_size)
print(f"Resized images saved to: {output_folder_path}")
