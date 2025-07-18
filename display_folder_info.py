import os
import datetime


def display_folder_info(directory_path):
    """
    Displays the total count of files and subfolders in the given directory.
    For each file, it prints the file name, creation date, and size on disk on a single line.
    
    Parameters:
        directory_path (str): The path to the directory.
    """
    # Verify that the path exists and is a directory.
    if not os.path.exists(directory_path):
        print(f"Error: The path '{directory_path}' does not exist.")
        return
    if not os.path.isdir(directory_path):
        print(f"Error: The path '{directory_path}' is not a directory.")
        return
    
    # Retrieve a list of file names and subfolders.
    file_list = [f for f in os.listdir(directory_path)
                 if os.path.isfile(os.path.join(directory_path, f))]
    subfolder_list = [f for f in os.listdir(directory_path)
                      if os.path.isdir(os.path.join(directory_path, f))]
    
    # Display the total counts.
    total_files = len(file_list)
    total_subfolders = len(subfolder_list)
    print(f"Total count of subfolders: {total_subfolders}")
    print(f"Total count of files: {total_files}")    
        
    # For each file, display its details on a single line.
    for file_name in file_list:
        file_path = os.path.join(directory_path, file_name)
        
        # Get creation time and convert it to a human-readable format.
        creation_time = os.path.getctime(file_path)
        creation_date = datetime.datetime.fromtimestamp(creation_time)
        creation_date_str = creation_date.strftime('%Y-%m-%d %H:%M:%S')
        
        # Get file size in bytes.
        file_size = os.path.getsize(file_path)
        
        # Print file information on a single line.
        print(f"{file_name} | {creation_date_str} | {file_size} bytes")