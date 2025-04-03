import os

def set_orientation(orientation, width, height):
    """Set the orientation of the image."""
    if orientation == "Landscape" and width > height:
        return width, height
    else:
        return height, width