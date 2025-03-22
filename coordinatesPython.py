import cv2
import numpy as np

def visualize_cells_and_save(cell_coordinates, output_path, image_path=None, image_shape=None, show_coordinates=False):
    """
    Visualizes cell coordinates on an image and saves the result.
    Optionally displays cell coordinates or cell numbers within each cell.

    Args:
        cell_coordinates: A list of tuples or dictionaries representing cell coordinates.
                          Tuples should be in the format (x1, y1, x2, y2).
                          Dictionaries should have keys 'x1', 'y1', 'x2', 'y2'.
        output_path: Path to save the output image.
        image_path: Path to the image file (optional). If provided, loads the image.
        image_shape: Tuple (width, height) specifying image dimensions (optional).
                     Required if image_path is not provided.
        show_coordinates: If True, displays cell coordinates within each cell.
                          If False, displays cell numbers.
    """

    if image_path:
        img = cv2.imread(image_path)
    elif image_shape:
        img = np.zeros((image_shape[1], image_shape[0], 3), dtype=np.uint8)
        img[:] = (255, 255, 255)  # Create a white image.
    else:
        raise ValueError("Either image_path or image_shape must be provided.")

    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    for i, cell in enumerate(cell_coordinates):
        if isinstance(cell, tuple):
            x1, y1, x2, y2 = cell
        elif isinstance(cell, dict):
            x1, y1, x2, y2 = cell['x1'], cell['y1'], cell['x2'], cell['y2']
        else:
            raise TypeError("Cell coordinates must be tuples or dictionaries.")

        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw green rectangles

        # Display cell number or coordinates within the cell
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        font_color = (255, 0, 0)  # Red color for text

        if show_coordinates:
            text = f"({x1},{y1})-({x2},{y2})"
        else:
            text = f"Cell {i}"

        text_x = x1 + 5
        text_y = y1 + 20

        cv2.putText(img, text, (text_x, text_y), font, font_scale, font_color, 1)

    cv2.imwrite(output_path, img)
    print(f"Image with cell information saved to: {output_path}")

# Example Usage:
cell_coordinates = [
    (108, 1213, 561, 1617), (561, 1213, 945, 1617), (945, 1213, 1390, 1617),
    (1390, 1213, 2000, 1617), (2000, 1213, 2361, 1617), (2361, 1213, 2361, 1617),
    (108, 1617, 561, 1875), (561, 1617, 945, 1875), (945, 1617, 1390, 1875),
    (2361, 1617, 2361, 3350), (108, 1875, 561, 1942), (561, 1875, 945, 1942),
    (945, 1875, 1390, 1942), (1390, 1875, 1390, 1942), (108, 1942, 561, 2073),
    (561, 1942, 945, 2073), (945, 1942, 1390, 2073), (1390, 1942, 2000, 2073),
    (2000, 1942, 2000, 2073), (108, 2073, 561, 2269), (561, 2073, 945, 2269),
    (945, 2073, 1390, 2269), (1390, 2073, 2000, 2269), (2000, 2073, 2000, 2269),
    (108, 2269, 561, 2465), (561, 2269, 945, 2465), (945, 2269, 1390, 2465),
    (1390, 2269, 2000, 2465), (2000, 2269, 2000, 2465), (108, 2465, 561, 2582),
    (561, 2465, 945, 2582), (945, 2465, 1390, 2582), (2000, 2465, 2000, 2715),
    (108, 2582, 561, 2649), (945, 2582, 1390, 2715), (1390, 2582, 1390, 2715),
    (108, 2649, 561, 2715), (561, 2649, 561, 2715), (108, 2715, 561, 3350),
    (561, 2715, 945, 3350), (945, 2715, 1390, 3350), (1390, 2715, 2000, 3350),
    (2000, 2715, 2000, 3350), (561, 3350, 945, 3350), (945, 3350, 1390, 3350),
    (1390, 3350, 2000, 3350), (2000, 3350, 2361, 3350), (2361, 3350, 2361, 3350)
]

# Provide image path or image dimensions and output path.
visualize_cells_and_save(
    cell_coordinates,
    "cells_with_info.png",
    image_shape=(2361, 3350),
    show_coordinates=True #Change to False to display Cell number.
)
visualize_cells_and_save(cell_coordinates, "cells_with_info.png", image_path="C:\\Indhu\\AI\\OCR Calude\\data\\output\\table_mask_2.png", show_coordinates=True) # if you have an image.