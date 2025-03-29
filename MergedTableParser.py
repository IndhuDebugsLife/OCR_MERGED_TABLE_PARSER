import os
import argparse
from pdf2image import convert_from_path
import pytesseract
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import itertools

def convert_pdf_to_images(pdf_path, output_folder, dpi=300):
    """
    Convert PDF pages to images
    
    Args:
        pdf_path: Path to the PDF file
        output_folder: Folder to save the image files
        dpi: Image resolution (dots per inch)
        
    Returns:
        List of paths to the saved images
    """
    print("here")
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Convert PDF to images
    print(f"Converting PDF: {pdf_path} to images...")
    images = convert_from_path(pdf_path, dpi=dpi)
    
    # Save images
    image_paths = []
    for i, image in enumerate(images):
        image_path = os.path.join(output_folder, f"page_{i+1}.png")
        image.save(image_path, "PNG")
        image_paths.append(image_path)
        
    print(f"Saved {len(image_paths)} images to {output_folder}")
    return image_paths

def preprocess_image(image_path):
    """
    Preprocess the image for better OCR and table detection
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Preprocessed image (OpenCV format)
    """
    # Read the image
    img = cv2.imread(image_path)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Adaptive thresholding to handle varying light conditions
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Alternatively, use Otsu's method
    # _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    return img, gray, binary

def detect_table_structure(binary_img):

    """
    Detect table structure including merged cells
    
    Args:
        binary_img: Binarized image
        
    Returns:
        Horizontal and vertical lines, and the image with detected lines
    """
    # Create copies of the binary image
    horizontal = binary_img.copy()
    vertical = binary_img.copy()
    
    # Specify size on horizontal axis
    cols = horizontal.shape[1]
    horizontal_size = cols // 30
    
    # Create structure element for extracting horizontal lines
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
    
    # Apply morphology operations
    horizontal = cv2.erode(horizontal, horizontalStructure)
    horizontal = cv2.dilate(horizontal, horizontalStructure)
    
    # Specify size on vertical axis
    rows = vertical.shape[0]
    vertical_size = rows // 70
    
    # Create structure element for extracting vertical lines
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_size))
    
    # Apply morphology operations
    vertical = cv2.erode(vertical, verticalStructure)
    vertical = cv2.dilate(vertical, verticalStructure)
    
    # Create a mask which includes the tables
    mask = horizontal + vertical
    
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours by area (largest first)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    return horizontal, vertical, mask, contours

def preprocess_image_gem(image_path):
    """
    Preprocess the image and print shapes and types.
    """
    # Read the image
    img = cv2.imread(image_path)
    print(f"Original Image (img) - Shape: {img.shape}, Type: {type(img)}, Data Type: {img.dtype}")

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(f"Grayscale Image (gray) - Shape: {gray.shape}, Type: {type(gray)}, Data Type: {gray.dtype}")

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive thresholding to handle varying light conditions
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    print(f"Binary Image (binary) - Shape: {binary.shape}, Type: {type(binary)}, Data Type: {binary.dtype}")

    return img, gray, binary

def detect_line_segments(horizontal, vertical):
    """
    Detect all horizontal and vertical line segments
    
    Args:
        horizontal: Horizontal lines image
        vertical: Vertical lines image
        
    Returns:
        Lists of horizontal and vertical line segments
    """
    # Detect horizontal line segments
    h_lines = cv2.HoughLinesP(
        horizontal, 
        rho=1, 
        theta=np.pi/180, 
        threshold=50, 
        minLineLength=horizontal.shape[1]//30,
        maxLineGap=20
    )
    
    # Detect vertical line segments
    v_lines = cv2.HoughLinesP(
        vertical, 
        rho=1, 
        theta=np.pi/180, 
        threshold=50, 
        minLineLength=vertical.shape[0]//10, 
        maxLineGap=20
    )
    
    # Extract and clean up horizontal lines
    h_segments = []
    if h_lines is not None:
        for line in h_lines:
            x1, y1, x2, y2 = line[0]
            # Ensure it's actually horizontal (small y difference)
            if abs(y2 - y1) < 10:
                h_segments.append((min(x1, x2), y1, max(x1, x2), y2))
    
    # Extract and clean up vertical lines
    v_segments = []
    if v_lines is not None:
        for line in v_lines:
            x1, y1, x2, y2 = line[0]
            # Ensure it's actually vertical (small x difference)
            if abs(x2 - x1) < 10:
                v_segments.append((x1, min(y1, y2), x2, max(y1, y2)))
    # Print detected segments
    print("Horizontal Line Segments:")
    for segment in h_segments:
        print(segment)
    
    print("\nVertical Line Segments:")
    for segment in v_segments:
        print(segment)
    return h_segments, v_segments

def cluster_points(points, axis=0, tolerance=15):
    """
    Cluster points along a specific axis
    
    Args:
        points: List of points (x, y)
        axis: Axis to cluster on (0 for x, 1 for y)
        tolerance: Maximum distance between points in the same cluster
        
    Returns:
        Dictionary mapping cluster centers to lists of points
    """
    if not points:
        return {}
        
    # Sort points by the specified axis
    sorted_points = sorted(points, key=lambda p: p[axis])
    
    clusters = {}
    current_value = sorted_points[0][axis]
    current_cluster = [sorted_points[0]]
    
    for point in sorted_points[1:]:
        if abs(point[axis] - current_value) <= tolerance:
            current_cluster.append(point)
        else:
            # Calculate average position for the cluster
            avg_pos = sum(p[axis] for p in current_cluster) // len(current_cluster)
            clusters[avg_pos] = current_cluster
            
            # Start a new cluster
            current_value = point[axis]
            current_cluster = [point]
    
    # Add the last cluster
    if current_cluster:
        avg_pos = sum(p[axis] for p in current_cluster) // len(current_cluster)
        clusters[avg_pos] = current_cluster
    
    return clusters

def identify_cells_from_grid(intersections, img_shape, h_segments, v_segments, tolerance=11):
    """
    Identify cells from a grid of intersections, handling merged cells with boundary verification and expansion.

    Args:
        intersections: List of intersection points (x, y)
        img_shape: Shape of the original image (height, width)
        h_segments: List of horizontal line segments [(x1, y1, x2, y2), ...].
        v_segments: List of vertical line segments [(x1, y1, x2, y2), ...].
        tolerance: Pixel tolerance for point clustering and boundary verification.

    Returns:
        List of cell coordinates (x1, y1, x2, y2) including merged cells
    """
    print("--- identify_cells_from_grid ---")
    print(f"Number of intersections: {len(intersections)}")

    # Cluster points by y-coordinate (rows)
    print("Clustering points by y-coordinate (rows)")
    row_clusters = cluster_points(intersections, axis=1, tolerance=tolerance)
    row_positions = sorted(row_clusters.keys())
    print(f"Row positions: {row_positions}")

    # Cluster points by x-coordinate (columns)
    print("Clustering points by x-coordinate (columns)")
    col_clusters = cluster_points(intersections, axis=0, tolerance=tolerance)
    col_positions = sorted(col_clusters.keys())
    print(f"Column positions: {col_positions}")

    # Create a 2D grid to represent the presence of intersection points
    print("Creating the grid")
    grid = np.zeros((len(row_positions), len(col_positions)), dtype=bool)

    # Map from grid coordinates to pixel coordinates
    print("Creating row and column maps")
    row_map = {i: pos for i, pos in enumerate(row_positions)}
    col_map = {i: pos for i, pos in enumerate(col_positions)}
    print(f"Row map: {row_map}")
    print(f"Column map: {col_map}")

    # Fill in the grid with detected intersections
    print("Filling the grid with intersections")
    for x, y in intersections:
        row_idx = min(range(len(row_positions)), key=lambda i: abs(row_positions[i] - y))
        col_idx = min(range(len(col_positions)), key=lambda i: abs(col_positions[i] - x))
        grid[row_idx, col_idx] = True
        print(f"Intersection at (x={x}, y={y}) mapped to grid[{row_idx}, {col_idx}]")

    # Identify all cells including merged cells
    print("Identifying cells")
    cells = []
    merged_cells = []

    
    for row_start in range(len(row_positions) - 1):
        for col_start in range(len(col_positions) - 1):
            if grid[row_start, col_start]:  # Check if a cell starts here
                
                # Find next horizontal intersection (col_end)
                col_end = None
                for c in range(col_start + 1, len(col_positions)):
                    if grid[row_start, c]:
                        col_end = c
                        break

                # Find next vertical intersection (row_end)
                row_end = None
                for r in range(row_start + 1, len(row_positions)):
                    if grid[r, col_start]:
                        row_end = r
                        break

                # Create cell if both intersections are found
                if col_end is not None and row_end is not None:
                    top_left = (col_map[col_start], row_map[row_start])
                    bottom_right = (col_map[col_end], row_map[row_end])
                    #cells.append((top_left[0], top_left[1], bottom_right[0], bottom_right[1]))
                    cell = (top_left[0], top_left[1], bottom_right[0], bottom_right[1])
                  #  print(f"Cell found: {cells[-1]}") check here to see cells right away
                #else:
                 #   print(f"Cell start at grid[{row_start}, {col_start}] but could not find both horizontal and vertical intersections.")
                    
                
                # Verify cell boundaries and expand if necessary
                if verify_cell_boundaries(cell, h_segments, v_segments, tolerance):
                    cells.append(cell)
                    print(f"  Cell added: (x1={cell[0]}, y1={cell[1]}, x2={cell[2]}, y2={cell[3]})")
                else:
                    print(f"  Cell boundaries not verified, attempting expansion...")
                    cell = expand_cell_boundaries(cell, h_segments, v_segments, row_positions, col_positions, tolerance)
                    if cell:
                        merged_cells.append(cell)
                        print(f"  Expanded cell added: (x1={cell[0]}, y1={cell[1]}, x2={cell[2]}, y2={cell[3]})")
                    else:
                        print("  Expansion failed.")
            else:
                print(f"  No valid bottom-right corner found.") 

    print("Cell identification complete.")
    for i, (x1, y1, x2, y2) in enumerate(cells):
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(f"Cell {i}: Coordinates ({x1}, {y1}, {x2}, {y2})")
    for i, (x1, y1, x2, y2) in enumerate(merged_cells):
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(f"Merged cells {i}: Coordinates ({x1}, {y1}, {x2}, {y2})")
    return cells,merged_cells

def expand_cell_boundaries(cell, h_segments, v_segments, row_positions, col_positions, tolerance):
    """Expands the cell boundaries to the bottom, stopping only when a valid boundary is found."""
    x1, y1, x2, y2 = cell
    original_cell = cell

    print(f"Debug: Expanding cell from ({x1}, {y1}, {x2}, {y2})")

    # Attempt Bottom Expansion
    y_index = row_positions.index(y2)
    
    while y_index < len(row_positions) - 1:
        y_index += 1
        y2 = row_positions[y_index]
        expanded_cell = (x1, original_cell[1], x2, y2)
        
        print(f"Debug: Trying bottom expansion to ({x1}, {original_cell[1]}, {x2}, {y2})")
        
        if verify_cell_boundaries(expanded_cell, h_segments, v_segments, tolerance):
            print(f"Debug: Expanded cell ({x1}, {original_cell[1]}, {x2}, {y2}) is valid.")
            print(f"Debug: Bottom expansion complete. Valid cell found. Returning.")
            return expanded_cell  # Return immediately when a valid cell is found.

    print(f"Debug: Bottom expansion complete. No valid expanded cell found.")
    return None  # Return None if no valid expansion is found.

def verify_cell_boundaries(cell, h_segments, v_segments, tolerance=11):
    """
    Verifies if the given cell boundaries are properly defined by the horizontal and vertical line segments.

    Args:
        cell: A tuple representing the cell boundaries (x1, y1, x2, y2).
        h_segments: List of horizontal line segments [(x1, y1, x2, y2), ...].
        v_segments: List of vertical line segments [(x1, y1, x2, y2), ...].
        tolerance: Maximum distance (in pixels) allowed between a boundary and a line segment.

    Returns:
        True if all four boundaries are verified, False otherwise.
    """
    x1, y1, x2, y2 = cell

    top_boundary_found = False
    bottom_boundary_found = False
    left_boundary_found = False
    right_boundary_found = False

    print(f"  Debug: Verifying cell boundaries for cell: {cell}")

    # 1. Verify Top Boundary
    print("    Debug: Checking top boundary...")
    for h_x1, h_y1, h_x2, h_y2 in h_segments:
        y_diff = abs(h_y1 - y1)
        x_overlap = max(0, min(x2, h_x2) - max(x1, h_x1))
        segment_length = abs(h_x2 - h_x1)
        cell_width = abs(x2 - x1)

        print(f"      Debug: Horizontal line: {(h_x1, h_y1, h_x2, h_y2)}")
        print(f"      Debug:   y_diff: {y_diff}, x_overlap: {x_overlap}, segment_length: {segment_length}, cell_width: {cell_width}")

        if y_diff <= tolerance and x_overlap > 2 and segment_length >= cell_width - tolerance:
            top_boundary_found = True
            print("      Debug:   Top boundary found.")
            break
        else:
            print("      Debug:   Top boundary not found for this line.")
 
    # 2. Verify Bottom Boundary
 
    print("    Debug: Checking bottom boundary...")
    for h_x1, h_y1, h_x2, h_y2 in h_segments:
        y_diff = abs(h_y1 - y2)
        x_overlap = max(0, min(x2, h_x2) - max(x1, h_x1))
        segment_length = abs(h_x2 - h_x1)
        cell_width = abs(x2 - x1)

        print(f"      Debug: Horizontal line: {(h_x1, h_y1, h_x2, h_y2)}")
        print(f"      Debug:   y_diff: {y_diff}, x_overlap: {x_overlap}, segment_length: {segment_length}, cell_width: {cell_width}")

        if y_diff <= tolerance and x_overlap > 2 and segment_length >= cell_width - tolerance:
            bottom_boundary_found = True
            print("      Debug:   Bottom boundary found.")
            break
        else:
            print("      Debug:   Bottom boundary not found for this line.")

    # 3. Verify Left Boundary
    print("    Debug: Checking left boundary...")
    for v_x1, v_y1, v_x2, v_y2 in v_segments:
        x_diff = abs(v_x1 - x1)
        y_overlap = max(0, min(y2, v_y2) - max(y1, v_y1))
        segment_length = abs(v_y2 - v_y1)
        cell_height = abs(y2 - y1)

        print(f"      Debug:   Vertical line: {(v_x1, v_y1, v_x2, v_y2)}")
        print(f"      Debug:   x_diff: {x_diff}, y_overlap: {y_overlap}, segment_length: {segment_length}, cell_height: {cell_height}")

        if x_diff <= tolerance and y_overlap > 5 and segment_length >= cell_height - tolerance:
            left_boundary_found = True
            print("      Debug:   Left boundary found.")
            break
        else:
           print("      Debug:   Left boundary not found for this line.")

    # 4. Verify Right Boundary
    print("    Debug: Checking right boundary...")
    for v_x1, v_y1, v_x2, v_y2 in v_segments:
        x_diff = abs(v_x2 - x2)
        y_overlap = max(0, min(y2, v_y2) - max(y1, v_y1))
        segment_length = abs(v_y2 - v_y1)
        cell_height = abs(y2 - y1)

        print(f"      Debug:   Vertical line: {(v_x1, v_y1, v_x2, v_y2)}")
        print(f"      Debug:   x_diff: {x_diff}, y_overlap: {y_overlap}, segment_length: {segment_length}, cell_height: {cell_height}")

        if x_diff <= tolerance and y_overlap > 5 and segment_length >= cell_height - tolerance:
            right_boundary_found = True
            print("      Debug:   Right boundary found.")
            break
        else:
          print("      Debug:   Right boundary not found for this line.")

    result = top_boundary_found and bottom_boundary_found and left_boundary_found and right_boundary_found

    # Determine which boundaries were not found
    not_found_boundaries = []
    if not result:
        if not top_boundary_found:
            not_found_boundaries.append("Top")
        if not bottom_boundary_found:
            not_found_boundaries.append("Bottom")
        if not left_boundary_found:
            not_found_boundaries.append("Left")
        if not right_boundary_found:
            not_found_boundaries.append("Right")


    print(f"    Debug: Verification result: {result}, Not found boundaries: {not_found_boundaries}")

    return result

def extract_text_from_cells(cells, gray_img, original_img):
    """
    Extract text from each cell using OCR, handling merged cells
    
    Args:
        cells: List of cell coordinates (x1, y1, x2, y2)
        gray_img: Grayscale image for OCR
        original_img: Original image for visualization
        
    Returns:
        List of dictionaries containing cell coordinates and extracted text,
        and an image showing the detected cells and extracted text
    """
    cell_data = []
    img_with_cells = original_img.copy()
    
    for i, (x1, y1, x2, y2) in enumerate(cells):
        # Ensure coordinates are valid
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Add a small margin inside the cell for better text extraction
        margin = 2
        x1_inner = x1 + margin
        y1_inner = y1 + margin
        x2_inner = x2 - margin
        y2_inner = y2 - margin
        
        # Ensure coordinates are within image bounds
        height, width = gray_img.shape
        x1_inner = max(0, min(x1_inner, width - 1))
        y1_inner = max(0, min(y1_inner, height - 1))
        x2_inner = max(0, min(x2_inner, width - 1))
        y2_inner = max(0, min(y2_inner, height - 1))
        
        # Skip invalid cells
        if x2_inner <= x1_inner or y2_inner <= y1_inner:
            continue
        
        # Draw the cell on the image
        cell_color = (0, 255, 0)  # Green for normal cells
        cell_thickness = 2
        
        # Check if this might be a merged cell (significantly larger than average)
        cell_width = x2 - x1
        cell_height = y2 - y1
        if cell_width > 100 or cell_height > 40:  # Adjust thresholds as needed
            # Likely a merged cell - use a different color
            cell_color = (0, 165, 255)  # Orange for merged cells
            cell_thickness = 3
        
        cv2.rectangle(img_with_cells, (x1, y1), (x2, y2), cell_color, cell_thickness)
        
        # Extract the cell region for OCR
        try:
            cell_img = gray_img[y1_inner:y2_inner, x1_inner:x2_inner]
            
            # Skip empty cells
            if cell_img.size == 0 or np.mean(cell_img) > 240:  # Skip if mostly white
                continue
            
            # Apply additional preprocessing for better OCR
            _, cell_binary = cv2.threshold(cell_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Apply morphological operations to clean up the image
            kernel = np.ones((2, 2), np.uint8)
            cell_binary = cv2.morphologyEx(cell_binary, cv2.MORPH_OPEN, kernel)
            
            # Convert to PIL Image for Tesseract
            pil_img = Image.fromarray(cell_binary)
            
            # Extract text using OCR
            # PSM modes: 6 = Assume a single uniform block of text, 4 = Assume a single column of text
            config = '--psm 6 --oem 3'
            text = pytesseract.image_to_string(pil_img, config=config)
            text = text.strip()
            
            # Add the text to the image
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.4
            font_color = (255, 0, 0)  # Red color for text overlay
            cv2.putText(img_with_cells, f"Cell {i}: {text[:10]}...", 
                        (x1, y1 - 5), font, font_scale, font_color, 1)
            
            # Store cell data with information about merged status
            is_merged_horiz = cell_width > 100  # Adjust threshold as needed
            is_merged_vert = cell_height > 40   # Adjust threshold as needed
            
            cell_data.append({
                'id': i,
                'x1': x1,
                'y1': y1,
                'x2': x2,
                'y2': y2,
                'width': cell_width,
                'height': cell_height,
                'text': text,
                'is_merged_horiz': is_merged_horiz,
                'is_merged_vert': is_merged_vert,
                'row': None,  # To be determined
                'col': None   # To be determined
            })
        except Exception as e:
            print(f"Warning: Error processing cell {i}: {str(e)}")
    
    return cell_data, img_with_cells

def find_grid_intersections(h_segments, v_segments, tolerance=5):
    """
    Find all grid intersections from horizontal and vertical line segments
    
    Args:
        h_segments: List of horizontal line segments (x1, y1, x2, y2)
        v_segments: List of vertical line segments (x1, y1, x2, y2)
        tolerance: Pixel tolerance for intersection detection
        
    Returns:
        List of intersection points (x, y)
    """
    intersections = []
    
    for h_x1, h_y1, h_x2, h_y2 in h_segments:
        h_y = (h_y1 + h_y2) // 2  # Average y-coordinate for horizontal line
        
        for v_x1, v_y1, v_x2, v_y2 in v_segments:
            v_x = (v_x1 + v_x2) // 2  # Average x-coordinate for vertical line
            
            # Check if vertical line spans this y-coordinate
            if v_y1 - tolerance <= h_y <= v_y2 + tolerance:
                # Check if horizontal line spans this x-coordinate
                if h_x1 - tolerance <= v_x <= h_x2 + tolerance:
                    intersections.append((v_x, h_y))
    
    return intersections

def extract_text_from_cells_final(cells,merged_cells, gray_img, original_img ):
    """
    Extract text from each cell using OCR, handling merged cells with grid analysis.
    Uses separate 'cells' and 'merged_cells' lists.
    """
    cell_data = []
    img_with_cells = original_img.copy()

    # Determine cell accessors based on cell type
    if isinstance(cells[0], tuple):
        get_x1 = lambda cell: cell[0]
        get_y1 = lambda cell: cell[1]
        get_x2 = lambda cell: cell[2]
        get_y2 = lambda cell: cell[3]
    else:
        get_x1 = lambda cell: cell['x1']
        get_y1 = lambda cell: cell['y1']
        get_x2 = lambda cell: cell['x2']
        get_y2 = lambda cell: cell['y2']

    # Process regular cells
    for i, cell in enumerate(cells):
        x1, y1, x2, y2 = get_x1(cell), get_y1(cell), get_x2(cell), get_y2(cell)
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # Apply margin and bounds check
        margin = 2
        x1_inner = max(0, min(x1 + margin, gray_img.shape[1] - 1))
        y1_inner = max(0, min(y1 + margin, gray_img.shape[0] - 1))
        x2_inner = max(0, min(x2 - margin, gray_img.shape[1] - 1))
        y2_inner = max(0, min(y2 - margin, gray_img.shape[0] - 1))

        if x2_inner <= x1_inner or y2_inner <= y1_inner:
            continue

        cv2.rectangle(img_with_cells, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green for normal cells

        # OCR and data storage
        try:
            cell_img = gray_img[y1_inner:y2_inner, x1_inner:x2_inner]
            if cell_img.size == 0 or np.mean(cell_img) > 260:
                continue

            _, cell_binary = cv2.threshold(cell_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            cell_binary = cv2.morphologyEx(cell_binary, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
            text = pytesseract.image_to_string(Image.fromarray(cell_binary), config='--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz₹').strip()

            cv2.putText(img_with_cells, f"Cell {i}: {text[:10]}...", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

            cell_data.append({
                'id': i,
                'x1': x1,
                'y1': y1,
                'x2': x2,
                'y2': y2,
                'width': x2 - x1,
                'height': y2 - y1,
                'text': text,
                'is_merged_horiz': False, #Normal cells are not merged
                'is_merged_vert': False, #Normal cells are not merged
                'row': None,
                'col': None
            })
            print(f"Cell {i}: x1 {x1}, y1 {y1}, x2 {x2}, y2 {y2}")
            print(f"Cell {i}: Width {x2 - x1}, Height {y2 - y1}, Merged Horiz: False, Merged Vert: False")
            print(f"Cell {i}: Tesseract Output '{text}'")

        except Exception as e:
            print(f"Warning: Error processing cell {i}: {str(e)}")

    # Process merged cells, if any
    if merged_cells:
        for i, cell in enumerate(merged_cells, start=len(cells)):
            x1, y1, x2, y2 = get_x1(cell), get_y1(cell), get_x2(cell), get_y2(cell)
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            margin = 2
            x1_inner = max(0, min(x1 + margin, gray_img.shape[1] - 1))
            y1_inner = max(0, min(y1 + margin, gray_img.shape[0] - 1))
            x2_inner = max(0, min(x2 - margin, gray_img.shape[1] - 1))
            y2_inner = max(0, min(y2 - margin, gray_img.shape[0] - 1))

            if x2_inner <= x1_inner or y2_inner <= y1_inner:
                continue

            cv2.rectangle(img_with_cells, (x1, y1), (x2, y2), (0, 165, 255), 3)  # Orange for merged cells

            try:
                cell_img = gray_img[y1_inner:y2_inner, x1_inner:x2_inner]
                if cell_img.size == 0 or np.mean(cell_img) > 260:
                    continue

                _, cell_binary = cv2.threshold(cell_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                cell_binary = cv2.morphologyEx(cell_binary, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
                text = pytesseract.image_to_string(Image.fromarray(cell_binary), config='--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz₹').strip()

                cv2.putText(img_with_cells, f"Merged Cell {i}: {text[:10]}...", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

                cell_data.append({
                    'id': i,
                    'x1': x1,
                    'y1': y1,
                    'x2': x2,
                    'y2': y2,
                    'width': x2 - x1,
                    'height': y2 - y1,
                    'text': text,
                    'is_merged_horiz': True, #Merged cells are merged.
                    'is_merged_vert': True, #Merged cells are merged.
                    'row': None,
                    'col': None
                })
                print(f"Merged Cell {i}: x1 {x1}, y1 {y1}, x2 {x2}, y2 {y2}")
                print(f"Merged Cell {i}: Width {x2 - x1}, Height {y2 - y1}, Merged Horiz: True, Merged Vert: True")
                print(f"Merged Cell {i}: Tesseract Output '{text}'")

            except Exception as e:
                print(f"Warning: Error processing merged cell {i}: {str(e)}")

    return cell_data, img_with_cells
    


def main():
    pdf_path ="./data/HDFCNew.pdf"
    images_folder ="./data/images"
    image_path = "./data/images/page_2.png"
    output_folder= "./data/output"
    i=1
   # image_paths=preprocess_image_gem( image_path)
    image_paths = convert_pdf_to_images(pdf_path, images_folder, 300)
    img, gray, binary = preprocess_image(image_path)
    horizontal, vertical, mask, contours = detect_table_structure(binary)
    cv2.imwrite(os.path.join(output_folder, f"horizontal_lines_{i+1}.png"), horizontal)
    cv2.imwrite(os.path.join(output_folder, f"vertical_lines_{i+1}.png"), vertical)
    cv2.imwrite(os.path.join(output_folder, f"table_mask_{i+1}.png"), mask)
    h_segments, v_segments = detect_line_segments(horizontal, vertical)
   
    
     # Draw line segments for debugging
    line_img = img.copy()
    for x1, y1, x2, y2 in h_segments: 
                cv2.line(line_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    for x1, y1, x2, y2 in v_segments:
                cv2.line(line_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.imwrite(os.path.join(output_folder, f"detected_lines_{i+1}.png"), line_img)
# Step 5: Find grid intersections
    intersections = find_grid_intersections(h_segments, v_segments)
            
    # Draw intersections for debugging
    intersection_img = img.copy()
    for x, y in intersections:
                cv2.circle(intersection_img, (x, y), 5, (0, 255, 255), -1)
    cv2.imwrite(os.path.join(output_folder, f"intersections_{i+1}.png"), intersection_img)


# Step 6: Identify cells from the grid
   
    cells,merged_cells=identify_cells_from_grid(intersections,img.shape[:2],h_segments,v_segments)
     
# Step 7: Extract text from cells
    cell_data, img_with_cells = extract_text_from_cells_final(cells,merged_cells, gray, img)
    cv2.imwrite(os.path.join(output_folder, f"detected_cells_new_{i+1}.png"), img_with_cells)
    
if __name__ == "__main__":
    main()