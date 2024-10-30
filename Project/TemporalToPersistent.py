import os
import shutil

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
        destination_path = os.path.join('./Project/', folder)
        os.makedirs(destination_path, exist_ok=True)

    folders_persistent = {
        'Persistent': 'Persistent'
    }

    destination_path = os.path.join('./Project/Landing Zone/', 'Persistent')
    os.makedirs(destination_path, exist_ok=True)


def move_files():

    # Folder paths
    source_folder = './Project/Landing Zone/Temporal'
    destination_path = './Project/Landing Zone/Persistent'

    # Create the folders and organize files automatically
    for filename in os.listdir(source_folder):
        file_path = os.path.join(source_folder, filename)
        if os.path.isfile(file_path):
            
            category_parts = filename.split('_')[:3]
            category = '_'.join(category_parts)
            destination_folder = os.path.join(destination_path, category)
            
            # Create the folder if not exists
            os.makedirs(destination_folder, exist_ok=True)
            
            # Move the file to the corresponding folder with error management
            try:
                shutil.copy(file_path, os.path.join(destination_folder, filename))
                # print(f"File '{filename}' moved to '{destination_folder}'")
            except Exception as e:
                print(f"Error to move the file '{filename}': {e}")

