import cv2
import numpy as np

def visualize_cells_with_specific_numbers(cell_data, output_path, image_path=None, image_shape=None):
    """
    Visualizes cells with their *specific* numbers as provided in the cell_data.

    Args:
        cell_data: A list of dictionaries, where each dictionary contains cell coordinates
                   and the 'id' of the cell. For example:
                   [
                       {'id': 1, 'x1': 561, 'y1': 1213, 'x2': 945, 'y2': 1617},
                       {'id': 2, 'x1': 945, 'y1': 1213, 'x2': 1390, 'y2': 1617},
                       ...
                   ]
        output_path: Path to save the output image.
        image_path: Path to the image file (optional). If provided, loads the image.
        image_shape: Tuple (width, height) specifying image dimensions (optional).
                     Required if image_path is not provided.
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

    for cell in cell_data:
        x1, y1, x2, y2 = int(cell['x1']), int(cell['y1']), int(cell['x2']), int(cell['y2'])
        cell_id = cell['id']  # Get the cell ID from the data

        # Draw rectangles
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)

        # Display cell ID
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_color = (255, 0, 0)
        font_thickness = 2
        text_x = x1 + 10
        text_y = y1 + 30
        cv2.putText(img, f"Cell {cell_id}", (text_x, text_y), font, font_scale, font_color, font_thickness)

    cv2.imwrite(output_path, img)
    print(f"Image with cell numbers saved to: {output_path}")

# Example Usage (using a list of dictionaries to define cells):
cell_data = [
    {'id': 1, 'x1': 561, 'y1': 1213, 'x2': 945, 'y2': 1617},
    {'id': 2, 'x1': 945, 'y1': 1213, 'x2': 1390, 'y2': 1617},
    {'id': 3, 'x1': 1390, 'y1': 1213, 'x2': 2000, 'y2': 1617},
    {'id': 4, 'x1': 2000, 'y1': 1213, 'x2': 2356, 'y2': 1617},
    {'id': 5, 'x1': 108, 'y1': 1617, 'x2': 561, 'y2': 1875},
    {'id': 6, 'x1': 561, 'y1': 1617, 'x2': 945, 'y2': 1875},
    {'id': 7, 'x1': 945, 'y1': 1617, 'x2': 1390, 'y2': 1875},
    {'id': 8, 'x1': 108, 'y1': 1875, 'x2': 561, 'y2': 1942},
    {'id': 9, 'x1': 561, 'y1': 1875, 'x2': 945, 'y2': 1942},
    {'id': 10, 'x1': 945, 'y1': 1875, 'x2': 1390, 'y2': 1942},
    {'id': 11, 'x1': 108, 'y1': 1942, 'x2': 561, 'y2': 2073},
    {'id': 12, 'x1': 561, 'y1': 1942, 'x2': 945, 'y2': 2073},
    {'id': 13, 'x1': 945, 'y1': 1942, 'x2': 1390, 'y2': 2073},
    {'id': 14, 'x1': 1390, 'y1': 1942, 'x2': 2000, 'y2': 2073},
    {'id': 15, 'x1': 108, 'y1': 2073, 'x2': 561, 'y2': 2269},
    {'id': 16, 'x1': 561, 'y1': 2073, 'x2': 945, 'y2': 2269},
    {'id': 17, 'x1': 945, 'y1': 2073, 'x2': 1390, 'y2': 2269},
    {'id': 18, 'x1': 1390, 'y1': 2073, 'x2': 2000, 'y2': 2269},
    {'id': 19, 'x1': 108, 'y1': 2269, 'x2': 561, 'y2': 2465},
    {'id': 20, 'x1': 561, 'y1': 2269, 'x2': 945, 'y2': 2465},
    {'id': 21, 'x1': 945, 'y1': 2269, 'x2': 1390, 'y2': 2465},
    {'id': 22, 'x1': 1390, 'y1': 2269, 'x2': 2000, 'y2': 2465},
    {'id': 23, 'x1': 108, 'y1': 2465, 'x2': 561, 'y2': 2582},
    {'id': 24, 'x1': 561, 'y1': 2465, 'x2': 945, 'y2': 2582},
    {'id': 25, 'x1': 945, 'y1': 2465, 'x2': 1390, 'y2': 2582},
    {'id': 26, 'x1': 108, 'y1': 2582, 'x2': 561, 'y2': 2649},
    {'id': 27, 'x1': 945, 'y1': 2582, 'x2': 1390, 'y2': 2715},
    {'id': 28, 'x1': 108, 'y1': 2649, 'x2': 561, 'y2': 2715},
    {'id': 29, 'x1': 108, 'y1': 2715, 'x2': 561, 'y2': 3350},
    {'id': 30, 'x1': 561, 'y1': 2715, 'x2': 945, 'y2': 3350},
    {'id': 31, 'x1': 945, 'y1': 2715, 'x2': 1390, 'y2': 3350},
    {'id': 32, 'x1': 1390, 'y1': 2715, 'x2': 2000, 'y2': 3350}
]

visualize_cells_with_specific_numbers(
    cell_data,
    "cells_with_correct_numbers.png",
    image_shape=(2361, 3350)
)

# If you have an image, you can use this instead:
# visualize_cells_with_specific_numbers(
#     cell_data,
#     "cells_with_correct_numbers.png",
#     image_path="path/to/your/image.png"
# )