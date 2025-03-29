import cv2
import numpy as np

def visualize_cell_and_segment(image_path, cell, segment, output_path="visualization.png"):
    """
    Visualizes a cell and a line segment on an image.

    Args:
        image_path: Path to the image file.
        cell: Tuple representing the cell boundaries (x1, y1, x2, y2).
        segment: Tuple representing the line segment (x1, y1, x2, y2).
        output_path: Path to save the visualized image.
    """
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    # Draw the cell
    x1_cell, y1_cell, x2_cell, y2_cell = map(int, cell)
    cv2.rectangle(img, (x1_cell, y1_cell), (x2_cell, y2_cell), (0, 255, 0), 2)  # Green rectangle

    # Draw the segment
    x1_seg, y1_seg, x2_seg, y2_seg = map(int, segment)
    cv2.line(img, (x1_seg, y1_seg), (x2_seg, y2_seg), (255, 0, 0), 2)  # Blue line

    # Save the image
    cv2.imwrite(output_path, img)
    print(f"Visualization saved to {output_path}")

if __name__ == "__main__":
    image_path = "./data/images/page_1.png"  # Replace with your image path
   # cell = (561, 2582, 945, 2649) 
    cell =(1390, 1617, 2000, 1942)
    segment = (118, 1944, 557, 1944)
    visualize_cell_and_segment(image_path, cell, segment)