#!/usr/bin/env python3
import json
import os
from pathlib import Path

def update_metadata():
    # Path to faces directory and metadata file
    faces_dir = Path("faces")
    metadata_file = Path("faces/metadata.json")
    
    # Check if faces directory exists
    if not faces_dir.exists():
        print(f"Error: {faces_dir} directory not found")
        return
    
    try:
        # Create metadata list
        metadata = []
        
        # Get all jpg files in faces directory
        for photo in faces_dir.glob("*.jpg"):
            # Get filename without extension
            filename = photo.stem
            
            # Check if it's a person's photo (ends with _0001)
            if filename.endswith("_0001"):
                # Extract person name (remove _0001)
                person_name = filename[:-5]
                
                # Create person entry
                person_entry = {
                    "name": person_name,
                    "photos": ["front"]  # Add front photo
                }
                
                # Add all other photos for this person
                for other_photo in faces_dir.glob(f"{person_name}_*.jpg"):
                    if other_photo.name != f"{person_name}_0001.jpg":
                        angle = other_photo.stem.split('_')[1]  # Get angle from filename
                        person_entry["photos"].append(angle)
                
                metadata.append(person_entry)
                print(f"Added {person_name} with {len(person_entry['photos'])} photos")
        
        # Save metadata
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Metadata created successfully with {len(metadata)} people")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    update_metadata() 