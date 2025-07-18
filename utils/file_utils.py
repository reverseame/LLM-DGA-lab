import os
import json

def save_to_text_file(filepath, data):
    """
    Saves data to a plain text file using JSON format.
    
    Args:
        filepath: Path to the file where data will be saved.
        data: Data to save.
    """
    with open(filepath, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4, ensure_ascii=False)

def load_from_text_file(filepath):
    """
    Loads data from a plain text file using JSON format.
    
    Args:
        filepath: Path to the file where data will be loaded from.
        
    Returns:
        The loaded data or None if the file does not exist.
    """
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as file:
            return json.load(file)
    return None