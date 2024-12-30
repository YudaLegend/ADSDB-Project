import os
import shutil


# The function to check if the substring contains number
def contains_number(substring):
    return any(char.isdigit() for char in substring)


def create_folders():
    # To create different zones taking into account that the landing zone has already been created
    # Dictionary of categories and destination folders
    folders_zone = {
        'Formatted Zone': 'Formatted Zone',
        'Trusted Zone': 'Trusted Zone',
        'Exploitation Zone': 'Exploitation Zone'
    }

    # Create the destination folders if not exist
    for folder in folders_zone.values():
        destination_path = os.path.join('./Data Management/', folder)
        os.makedirs(destination_path, exist_ok=True)

    destination_path = os.path.join('./Data Management/Landing Zone/', 'Persistent')
    os.makedirs(destination_path, exist_ok=True)


def move_files():

    # Folder paths
    source_folder = './Data Management/Landing Zone/Temporal'
    destination_path = './Data Management/Landing Zone/Persistent'
    try:
        # Create the folders and organize files automatically
        for filename in os.listdir(source_folder):

            # Get the filename directory
            file_path = os.path.join(source_folder, filename)

            # Check if the path is a file or not
            if os.path.isfile(file_path):
                
                # Split the filename into different parts
                category_parts = filename.split('_')

                # Extract the category from the filename by iterating until a number is found.
                category = []
                for i in category_parts:
                    if contains_number(i):
                        break
                    category.append(i)
                
                # Join the category parts to form the category name
                category = '_'.join(category)

                # Create the destination folder path if it doesn't already exist
                destination_folder = os.path.join(destination_path, category)
                os.makedirs(destination_folder, exist_ok=True)
                
                # Move the file to the corresponding folder with error management
                try:
                    shutil.copy(file_path, os.path.join(destination_folder, filename))
                    # print(f"File '{filename}' moved to '{destination_folder}'")
                except Exception as e:
                    print(f"Error to move the file '{filename}': {e}")
                    return False
    except Exception as e:
        print(f"Error to move the files: {e}")
        return False
    except FileNotFoundError:
        print(f"Folder not found")
        return False
    return True

