import os
import argparse
from pdf2image import convert_from_path
import pytesseract
import cv2
import numpy as np
import pandas as pd
from PIL import Image

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

def identify_cells_from_grid(intersections, img_shape, tolerance=15 ):
    """
    Identify cells from a grid of intersections, handling merged cells
    
    Args:
        intersections: List of intersection points (x, y)
        img_shape: Shape of the original image (height, width)
        tolerance: Pixel tolerance for point clustering
        
    Returns:
        List of cell coordinates (x1, y1, x2, y2) including merged cells
    """
    # Cluster points by y-coordinate (rows)
    row_clusters = cluster_points(intersections, axis=1, tolerance=tolerance)
    row_positions = sorted(row_clusters.keys())
    
    # Cluster points by x-coordinate (columns)
    col_clusters = cluster_points(intersections, axis=0, tolerance=tolerance)
    col_positions = sorted(col_clusters.keys())
    
    # Create a 2D grid to represent the presence of intersection points
    grid = np.zeros((len(row_positions), len(col_positions)), dtype=bool)
    
    # Map from grid coordinates to pixel coordinates
    row_map = {i: pos for i, pos in enumerate(row_positions)}
    col_map = {i: pos for i, pos in enumerate(col_positions)}
    
    # Fill in the grid with detected intersections
    for x, y in intersections:
        # Find the closest row and column positions
        row_idx = min(range(len(row_positions)), key=lambda i: abs(row_positions[i] - y))
        col_idx = min(range(len(col_positions)), key=lambda i: abs(col_positions[i] - x))
        
        # Mark this grid cell as having an intersection
        grid[row_idx, col_idx] = True
    
    # Identify all cells including merged cells
    cells = []
    
    # Scan the grid to find cells
    for row_start in range(len(row_positions) - 1):
        for col_start in range(len(col_positions) - 1):
            # Check if we have a valid cell starting point (top-left corner)
            if not grid[row_start, col_start]:
                continue
                
            # Find how far right this cell extends (handle horizontal merging)
            col_end = col_start
            for c in range(col_start + 1, len(col_positions)):
                if grid[row_start, c]:
                    col_end = c
                    break
            
            # Find how far down this cell extends (handle vertical merging)
            row_end = row_start
            for r in range(row_start + 1, len(row_positions)):
                if grid[r, col_start]:
                    row_end = r
                    break
            
            # Now check if we have a valid bottom-right corner
            if grid[row_end, col_end]:
                # Valid cell found
                top_left = (col_map[col_start], row_map[row_start])
                bottom_right = (col_map[col_end], row_map[row_end])
                
                # Add the cell
                cells.append((
                    top_left[0], top_left[1],          # x1, y1
                    bottom_right[0], bottom_right[1]   # x2, y2
                ))
            print(f"cell logic")
            for i, (x1, y1, x2, y2) in enumerate(cells):
        # Ensure coordinates are valid
               x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
               print(f"Cell {i}: Coordinates ({x1}, {y1}, {x2}, {y2})")
                # Optional: Mark this region as processed to avoid duplicate cells
                # Not strictly necessary but can be added for efficiency
    
    return cells




    """
    Identifies cells from a grid in an image with debug statements.

    Args:
        image_path: Path to the image file.
        clustering_tolerance: Tolerance for intersection point clustering.
    Returns:
        A list of cell coordinates (tuples) and the image with cell outlines.
    """

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # 1. Line Detection (Hough Transform)
    horizontal_lines = cv2.HoughLines(binary, 1, np.pi / 180, 200)
    vertical_lines = cv2.HoughLines(binary, 1, np.pi / 90, 200)

    print(f"Debug: Horizontal lines detected: {len(horizontal_lines) if horizontal_lines is not None else 0}")
    print(f"Debug: Vertical lines detected: {len(vertical_lines) if vertical_lines is not None else 0}")

    def get_line_coords(lines, is_horizontal):
        coords = []
        if lines is not None:
            for line in lines:
                rho, theta = line[0]
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b if is_horizontal else a))
                y1 = int(y0 + 1000 * (a if is_horizontal else -b))
                x2 = int(x0 - 1000 * (-b if is_horizontal else a))
                y2 = int(y0 - 1000 * (a if is_horizontal else -b))
                coords.append(((x1, y1), (x2, y2)))
        return coords

    horizontal_coords = get_line_coords(horizontal_lines, True)
    vertical_coords = get_line_coords(vertical_lines, False)

    # 2. Intersection Detection
    intersections = []
    for h_line in horizontal_coords:
        for v_line in vertical_coords:
            h_p1, h_p2 = h_line
            v_p1, v_p2 = v_line
            denom = (v_p2[0] - v_p1[0]) * (h_p2[1] - h_p1[1]) - (v_p2[1] - v_p1[1]) * (h_p2[0] - h_p1[0])
            if denom != 0:
                x = ((v_p2[0] - v_p1[0]) * (h_p1[0] * h_p2[1] - h_p1[1] * h_p2[0]) - (h_p2[0] - h_p1[0]) * (v_p1[0] * v_p2[1] - v_p1[1] * v_p2[0])) / denom
                y = ((v_p2[1] - v_p1[1]) * (h_p1[0] * h_p2[1] - h_p1[1] * h_p2[0]) - (h_p2[1] - h_p1[1]) * (v_p1[0] * v_p2[1] - v_p1[1] * v_p2[0])) / denom
                intersections.append((int(x), int(y)))

    print(f"Debug: Intersections detected: {len(intersections)}")

    # 3. Intersection Point Clustering
    clustered_intersections = []
    used_points = set()

    for i, point1 in enumerate(intersections):
        if i in used_points:
            continue
        cluster = [point1]
        for j, point2 in enumerate(intersections):
            if i != j and j not in used_points:
                dist = np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
                if dist < clustering_tolerance:
                    cluster.append(point2)
                    used_points.add(j)
        avg_x = int(np.mean([p[0] for p in cluster]))
        avg_y = int(np.mean([p[1] for p in cluster]))
        clustered_intersections.append((avg_x, avg_y))

    print(f"Debug: Clustered intersections: {len(clustered_intersections)}")

    # 4. Cell Formation
    cells = []
    clustered_intersections.sort()  # Sort for consistent cell formation
    for i in range(len(clustered_intersections) - 1):
        for j in range(i + 1, len(clustered_intersections)):
            x1, y1 = clustered_intersections[i]
            x2, y2 = clustered_intersections[j]
            if x1 < x2 and y1 < y2:
                # Check for other intersection points in between to ensure a valid cell
                valid_cell = True
                for k in range(len(clustered_intersections)):
                    xk, yk = clustered_intersections[k]
                    if x1 < xk < x2 and y1 < yk < y2:
                        # Check if any other point is within the cell.
                        if k not in (i, j):
                          valid_cell = False
                          break;
                if valid_cell:
                    cells.append((x1, y1, x2, y2))
    print(f"Debug: Cells formed: {len(cells)}")

    # Draw Cells
    img_with_cells = img.copy()
    for x1, y1, x2, y2 in cells:
        cv2.rectangle(img_with_cells, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return cells, img_with_cells



def identify_cells_from_grid_new(intersections, img_shape, tolerance=15):
    """
    Identify cells from a grid of intersections, handling merged cells with debug statements.

    Args:
        intersections: List of intersection points (x, y)
        img_shape: Shape of the original image (height, width)
        tolerance: Pixel tolerance for point clustering

    Returns:
        List of cell coordinates (x1, y1, x2, y2) including merged cells
    """
    print(f"Debug: Input intersections: {len(intersections)}")
    print(f"Debug: Image shape: {img_shape}")
    print(f"Debug: Clustering tolerance: {tolerance}")

    def cluster_points(points, axis, tolerance):
        clusters = {}
        used = set()
        for i, p1 in enumerate(points):
            if i in used:
                continue
            cluster = [p1]
            for j, p2 in enumerate(points):
                if i != j and j not in used:
                    dist = abs(p1[axis] - p2[axis])
                    if dist < tolerance:
                        cluster.append(p2)
                        used.add(j)
            avg = int(np.mean([p[axis] for p in cluster]))
            clusters[avg] = cluster
        return clusters

    # Cluster points by y-coordinate (rows)
    row_clusters = cluster_points(intersections, axis=1, tolerance=tolerance)
    row_positions = sorted(row_clusters.keys())

    print(f"Debug: Row clusters: {len(row_clusters)}")
    print(f"Debug: Row positions: {row_positions}")

    # Cluster points by x-coordinate (columns)
    col_clusters = cluster_points(intersections, axis=0, tolerance=tolerance)
    col_positions = sorted(col_clusters.keys())

    print(f"Debug: Column clusters: {len(col_clusters)}")
    print(f"Debug: Column positions: {col_positions}")

    # Create a 2D grid to represent the presence of intersection points
    grid = np.zeros((len(row_positions), len(col_positions)), dtype=bool)

    # Map from grid coordinates to pixel coordinates
    row_map = {i: pos for i, pos in enumerate(row_positions)}
    col_map = {i: pos for i, pos in enumerate(col_positions)}

    # Fill in the grid with detected intersections
    for x, y in intersections:
        # Find the closest row and column positions
        row_idx = min(range(len(row_positions)), key=lambda i: abs(row_positions[i] - y))
        col_idx = min(range(len(col_positions)), key=lambda i: abs(col_positions[i] - x))

        # Mark this grid cell as having an intersection
        grid[row_idx, col_idx] = True

    print(f"Debug: Grid filled. Shape: {grid.shape}")

    # Identify all cells including merged cells
    cells = []

    # Scan the grid to find cells
    for row_start in range(len(row_positions) - 1):
        for col_start in range(len(col_positions) - 1):
            # Check if we have a valid cell starting point (top-left corner)
            if not grid[row_start, col_start]:
                continue

            # Find how far right this cell extends (handle horizontal merging)
            col_end = col_start
            for c in range(col_start + 1, len(col_positions)):
                if grid[row_start, c]:
                    col_end = c
                    break

            # Find how far down this cell extends (handle vertical merging)
            row_end = row_start
            for r in range(row_start + 1, len(row_positions)):
                if grid[r, col_start]:
                    row_end = r
                    break

            # Now check if we have a valid bottom-right corner
            if grid[row_end, col_end]:
                # Valid cell found
                top_left = (col_map[col_start], row_map[row_start])
                bottom_right = (col_map[col_end], row_map[row_end])

                # Add the cell
                cells.append((
                    top_left[0], top_left[1],  # x1, y1
                    bottom_right[0], bottom_right[1]  # x2, y2
                ))

    # Print cell logic and coordinates only once after all cells are found
    print("Cell logic:")
    for i, (x1, y1, x2, y2) in enumerate(cells):
        # Ensure coordinates are valid
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(f"Cell {i}: Coordinates ({x1}, {y1}, {x2}, {y2})")

    print(f"Debug: Cells identified: {len(cells)}")
    return cells

def identify_cells_from_grid_neww(intersections, img_shape, h_segments, v_segments, tolerance=15):
    """
    Identify cells from a grid of intersections, handling merged cells,
    and verifying cell boundaries with line segments.

    Args:
        intersections: List of intersection points (x, y)
        img_shape: Shape of the original image (height, width)
        h_segments: List of horizontal line segments
        v_segments: List of vertical line segments
        tolerance: Pixel tolerance for point clustering

    Returns:
        List of cell coordinates (x1, y1, x2, y2) including merged cells
    """
    print(f"Debug: Input intersections: {len(intersections)}")
    print(f"Debug: Image shape: {img_shape}")
    print(f"Debug: Clustering tolerance: {tolerance}")

    def cluster_points(points, axis, tolerance):
        clusters = {}
        used = set()
        for i, p1 in enumerate(points):
            if i in used:
                continue
            cluster = [p1]
            for j, p2 in enumerate(points):
                if i != j and j not in used:
                    dist = abs(p1[axis] - p2[axis])
                    if dist < tolerance:
                        cluster.append(p2)
                        used.add(j)
            avg = int(np.mean([p[axis] for p in cluster]))
            clusters[avg] = cluster
        return clusters

    # Cluster points by y-coordinate (rows)
    row_clusters = cluster_points(intersections, axis=1, tolerance=tolerance)
    row_positions = sorted(row_clusters.keys())

    print(f"Debug: Row clusters: {len(row_clusters)}")
    print(f"Debug: Row positions: {row_positions}")

    # Cluster points by x-coordinate (columns)
    col_clusters = cluster_points(intersections, axis=0, tolerance=tolerance)
    col_positions = sorted(col_clusters.keys())

    print(f"Debug: Column clusters: {len(col_clusters)}")
    print(f"Debug: Column positions: {col_positions}")

    # Create a 2D grid to represent the presence of intersection points
    grid = np.zeros((len(row_positions), len(col_positions)), dtype=bool)

    # Map from grid coordinates to pixel coordinates
    row_map = {i: pos for i, pos in enumerate(row_positions)}
    col_map = {i: pos for i, pos in enumerate(col_positions)}

    # Fill in the grid with detected intersections
    for x, y in intersections:
        # Find the closest row and column positions
        row_idx = min(range(len(row_positions)), key=lambda i: abs(row_positions[i] - y))
        col_idx = min(range(len(col_positions)), key=lambda i: abs(col_positions[i] - x))

        # Mark this grid cell as having an intersection
        grid[row_idx, col_idx] = True

    print(f"Debug: Grid filled. Shape: {grid.shape}")

    # Identify all cells including merged cells
    cells =[]

    # Scan the grid to find cells
    for row_start in range(len(row_positions) - 1):
        for col_start in range(len(col_positions) - 1):
            # Check if we have a valid cell starting point (top-left corner)
            if not grid[row_start, col_start]:
                continue

            # Find how far right this cell extends (handle horizontal merging)
            col_end = col_start
            for c in range(col_start + 1, len(col_positions)):
                if grid[row_start, c]:
                    col_end = c
                    break

            # Find how far down this cell extends (handle vertical merging)
            row_end = row_start
            for r in range(row_start + 1, len(row_positions)):
                if grid[r, col_start]:
                    row_end = r
                    break

            # Now check if we have a valid bottom-right corner
            if grid[row_end, col_end]:
                # Valid cell found
                top_left = (col_map[col_start], row_map[row_start])
                bottom_right = (col_map[col_end], row_map[row_end])

                # Verify cell boundaries before adding the cell
                if verify_cell_boundaries(
                    (top_left[0], top_left[1], bottom_right[0], bottom_right[1]),
                    h_segments, v_segments, tolerance
                ):
                    # Add the cell
                    cells.append((
                        top_left[0], top_left[1],  # x1, y1
                        bottom_right[0], bottom_right[1]  # x2, y2
                    ))

    # Print cell logic and coordinates only once after all cells are found
    
    print("Cell logic:")
    for i, (x1, y1, x2, y2) in enumerate(cells):
        # Ensure coordinates are valid
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(f"Cell {i}: Coordinates ({x1}, {y1}, {x2}, {y2})")

    print(f"Debug: Cells identified: {len(cells)}")
    return cells

def verify_cell_boundaries_old(cell, h_segments, v_segments, tolerance):
    """
    Verifies if the cell boundaries are supported by the given line segments.

    Args:
        cell: A tuple representing the cell (x1, y1, x2, y2)
        h_segments: List of horizontal line segments
        v_segments: List of vertical line segments
        tolerance: Pixel tolerance for line segment verification

    Returns:
        True if all four boundaries are supported by line segments, False otherwise
    """
    x1, y1, x2, y2 = cell

    print(f"  Debug: Verifying cell boundaries for cell: {cell}")  # Added debug
    tolerance=5
  # Check for horizontal lines (top and bottom)
    top_boundary_found = False
    bottom_boundary_found = False
    for h_x1, h_y1, h_x2, h_y2 in h_segments:
        print(f"    Debug: ************ boundary line: {(h_x1, h_y1, h_x2, h_y2)}")
     
        # Check for top boundary
        x_diff_top = abs(h_x1 - x1)
        x_diff_bot = abs(h_x2 - x2)
        y_diff_top = abs(h_y1 - y1)
        y_diff_bot = abs(h_y2 - y2)
        length_line= abs(abs(x2 - x1)- abs(h_x2 - h_x1)) # Distance from line start to cell start
      
        print(f"    Debug: ************ boundary line: {(h_x1, h_y1, h_x2, h_y2)} ")
        #print(f"X point diff: {x_diff_top} Y plane top: {y_diff_bot} length_line: {y_diff_bot}")
    
        if x_diff_top < tolerance and y_diff_top < tolerance and\
              length_line >= 0:
            length_line = True
            print(f"    Debug: Found top boundary line: {(h_x1, h_y1, h_x2, h_y2)}")  # Added debug
            break
        if x_diff_bot < tolerance and y_diff_bot < tolerance and\
              length_line >= 0:
            bottom_boundary_found = True
            print(f"    Debug: Found bottom boundary line: {(h_x1, h_y1, h_x2, h_y2)}")  # Added debug
            break
    print(f"start")
    # Check for vertical lines (left and right)
    left_boundary_found = False
    right_boundary_found = False
    vx_diff_top = abs(v_x1 - x1)
    vx_diff_bot = abs(v_x2 - x2)
    vy_diff_top = abs(v_y1 - y1)
    vy_diff_bot = abs(v_y2 - y2)
    length_line= abs(abs(y2 - y1)- abs(v_y2 - v_y1)) # Distance from line start to cell start

    print(f"    Debug: ************ boundary line: {(h_x1, h_y1, h_x2, h_y2)} ")
   
    for v_x1, v_y1, v_x2, v_y2 in v_segments:
        print(f"    Debug: ##################################  boundary line: {(v_x1, v_y1, v_x2, v_y2)}")
        if vx_diff_top < tolerance and vy_diff_top <tolerance and length_line >0:
            left_boundary_found = True
            print(f"    Debug: Found left boundary line: {(v_x1, v_y1, v_x2, v_y2)}")  # Added debug
        if vx_diff_bot < tolerance and vy_diff_top <tolerance and length_line >0:
            right_boundary_found = True
            print(f"    Debug: Found right boundary line: {(v_x1, v_y1, v_x2, v_y2)}")  # Added debug

    result = top_boundary_found and bottom_boundary_found and \
           left_boundary_found  and right_boundary_found

    print(f"    Debug: Boundary verification result: {result}")  # Added debug
    return result   

def verify_cell_boundaries(cell, h_segments, v_segments, tolerance=5):
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

def visualize_verified_cells(image, cells):
    """
    Visualizes verified cell boundaries on an image.

    Args:
        image: The original image (OpenCV format).
        cells:  A list of cell coordinates, where each cell is a tuple
                (x1, y1, x2, y2).
    Returns:
        The image with the cell boundaries drawn.
    """
    img_with_cells = image.copy()  # Create a copy to draw on
    for x1, y1, x2, y2 in cells:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Ensure integers
        cv2.rectangle(img_with_cells, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw green rectangle
    return img_with_cells

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

def extract_text_from_cells_neww(cells, gray_img, original_img): 
    """
    Extract text from each cell using OCR, handling merged cells with grid analysis.
    Handles both tuple and dictionary cell inputs.
    """
    cell_data = []
    img_with_cells = original_img.copy()

    # Check if cells are tuples or dictionaries
    if isinstance(cells[0], tuple):
        # Cells are tuples
        widths = [cell[2] - cell[0] for cell in cells]
        heights = [cell[3] - cell[1] for cell in cells]
        get_x1 = lambda cell: cell[0]
        get_y1 = lambda cell: cell[1]
        get_x2 = lambda cell: cell[2]
        get_y2 = lambda cell: cell[3]

    else:
        # Cells are dictionaries
        widths = [cell['x2'] - cell['x1'] for cell in cells]
        heights = [cell['y2'] - cell['y1'] for cell in cells]
        get_x1 = lambda cell: cell['x1']
        get_y1 = lambda cell: cell['y1']
        get_x2 = lambda cell: cell['x2']
        get_y2 = lambda cell: cell['y2']

    avg_width = sum(widths) / len(widths) if widths else 0
    avg_height = sum(heights) / len(heights) if heights else 0

    # Define adaptive merge thresholds
    merged_width_threshold = avg_width * 1.5 if avg_width else 100
    merged_height_threshold = avg_height * 1.5 if avg_height else 40

    for i, cell in enumerate(cells):
        x1, y1, x2, y2 = get_x1(cell), get_y1(cell), get_x2(cell), get_y2(cell)

        # Ensure coordinates are valid
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # Add a small margin inside the cell for better text extraction
        margin = 2
        x1_inner = x1 + margin
        y1_inner = y1 + margin
        x2_inner = x2 - margin
        y2_inner = y2 - margin

        # Ensure coordinates are within image bounds
        height_img, width_img = gray_img.shape
        x1_inner = max(0, min(x1_inner, width_img - 1))
        y1_inner = max(0, min(y1_inner, height_img - 1))
        x2_inner = max(0, min(x2_inner, width_img - 1))
        y2_inner = max(0, min(y2_inner, height_img - 1))

        # Skip invalid cells
        if x2_inner <= x1_inner or y2_inner <= y1_inner:
            continue

        # Draw the cell on the image
        cell_color = (0, 255, 0)  # Green for normal cells
        cell_thickness = 2

        # Check if this might be a merged cell (using adaptive thresholds)
        cell_width = x2 - x1
        cell_height = y2 - y1
        is_merged_horiz = cell_width > merged_width_threshold
        is_merged_vert = cell_height > merged_height_threshold

        if is_merged_horiz or is_merged_vert:
            cell_color = (0, 165, 255)  # Orange for merged cells
            cell_thickness = 3

        cv2.rectangle(img_with_cells, (x1, y1), (x2, y2), cell_color, cell_thickness)

        # Extract the cell region for OCR
        try:
            cell_img = gray_img[y1_inner:y2_inner, x1_inner:x2_inner]

            # Skip empty cells and mostly white cells
            if cell_img.size == 0:
                print(f"Cell {i}: Skipped - Empty cell.")
                continue
            mean_intensity = np.mean(cell_img)
            if mean_intensity > 260:
                print(f"Cell {i}: Skipped - Mostly white cell (mean: {mean_intensity:.2f}).")
                continue

            # Apply additional preprocessing for better OCR
            _, cell_binary = cv2.threshold(cell_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Apply morphological operations to clean up the image
            kernel = np.ones((2, 2), np.uint8)
            cell_binary = cv2.morphologyEx(cell_binary, cv2.MORPH_OPEN, kernel)

            # Convert to PIL Image for Tesseract
            pil_img = Image.fromarray(cell_binary)

            # Extract text using OCR
            config = '--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz₹'
            text = pytesseract.image_to_string(pil_img, config=config)
            text = text.strip()

            # Add the text to the image
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.4
            font_color = (255, 0, 0)  # Red color for text overlay
            cv2.putText(img_with_cells, f"Cell {i}: {text[:10]}...",
                        (x1, y1 - 5), font, font_scale, font_color, 1)

            # Store cell data with information about merged status
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
                'col': None  # To be determined
            })
            #Debugging print statements.
            print(f"Cell {i}: x1 {x1}, y1 {y1}, x2 {x2}, y2 {y2}")
            print(f"Cell {i}: Width {cell_width}, Height {cell_height}, Merged Horiz: {is_merged_horiz}, Merged Vert: {is_merged_vert}")
            print(f"Cell {i}: Tesseract Output '{text}'")

        except Exception as e:
            print(f"Warning: Error processing cell {i}: {str(e)}")

    return cell_data, img_with_cells

def extract_text_from_cells_new(cells, gray_img, original_img):
    """
    Extract text from each cell using OCR, handling merged cells
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

            # Skip empty cells and mostly white cells
            if cell_img.size == 0:
                print(f"Cell {i}: Skipped - Empty cell.")
                continue
            mean_intensity = np.mean(cell_img)
            if mean_intensity > 260:
                print(f"Cell {i}: Skipped - Mostly white cell (mean: {mean_intensity:.2f}).")
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
            config = '--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz₹' #Added Char whitelist.
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
            is_merged_vert = cell_height > 40  # Adjust threshold as needed

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
                'col': None  # To be determined
            })
            #Debugging print statements.
            print(f"Cell {i}: Coordinates ({x1_inner}, {y1_inner}, {x2_inner}, {y2_inner})")
            print(f"Cell {i}: Size {cell_img.size}, Mean {np.mean(cell_img)}")
            print(f"Cell {i}: Tesseract Output '{text}'")

        except Exception as e:
            print(f"Warning: Error processing cell {i}: {str(e)}")

    return cell_data, img_with_cells


    """
    Extract text from each cell using OCR, handling merged cells with grid analysis.
    """
    cell_data = []
    img_with_cells = original_img.copy()

    # Calculate average cell dimensions
    widths = [cell['x2'] - cell['x1'] for cell in cells]
    heights = [cell['y2'] - cell['y1'] for cell in cells]
    avg_width = sum(widths) / len(widths) if widths else 0
    avg_height = sum(heights) / len(heights) if heights else 0

    # Define adaptive merge thresholds
    merged_width_threshold = avg_width * 1.5 if avg_width else 100
    merged_height_threshold = avg_height * 1.5 if avg_height else 40

    for i, cell in enumerate(cells):
        x1, y1, x2, y2 = cell['x1'], cell['y1'], cell['x2'], cell['y2']

        # Ensure coordinates are valid
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # Add a small margin inside the cell for better text extraction
        margin = 2
        x1_inner = x1 + margin
        y1_inner = y1 + margin
        x2_inner = x2 - margin
        y2_inner = y2 - margin

        # Ensure coordinates are within image bounds
        height_img, width_img = gray_img.shape
        x1_inner = max(0, min(x1_inner, width_img - 1))
        y1_inner = max(0, min(y1_inner, height_img - 1))
        x2_inner = max(0, min(x2_inner, width_img - 1))
        y2_inner = max(0, min(y2_inner, height_img - 1))

        # Skip invalid cells
        if x2_inner <= x1_inner or y2_inner <= y1_inner:
            continue

        # Draw the cell on the image
        cell_color = (0, 255, 0)  # Green for normal cells
        cell_thickness = 2

        # Check if this might be a merged cell (using adaptive thresholds)
        cell_width = x2 - x1
        cell_height = y2 - y1
        is_merged_horiz = cell_width > merged_width_threshold
        is_merged_vert = cell_height > merged_height_threshold

        if is_merged_horiz or is_merged_vert:
            cell_color = (0, 165, 255)  # Orange for merged cells
            cell_thickness = 3

        cv2.rectangle(img_with_cells, (x1, y1), (x2, y2), cell_color, cell_thickness)

        # Extract the cell region for OCR
        try:
            cell_img = gray_img[y1_inner:y2_inner, x1_inner:x2_inner]

            # Skip empty cells and mostly white cells
            if cell_img.size == 0:
                print(f"Cell {i}: Skipped - Empty cell.")
                continue
            mean_intensity = np.mean(cell_img)
            if mean_intensity > 240:
                print(f"Cell {i}: Skipped - Mostly white cell (mean: {mean_intensity:.2f}).")
                continue

            # Apply additional preprocessing for better OCR
            _, cell_binary = cv2.threshold(cell_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Apply morphological operations to clean up the image
            kernel = np.ones((2, 2), np.uint8)
            cell_binary = cv2.morphologyEx(cell_binary, cv2.MORPH_OPEN, kernel)

            # Convert to PIL Image for Tesseract
            pil_img = Image.fromarray(cell_binary)

            # Extract text using OCR
            config = '--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz₹'
            text = pytesseract.image_to_string(pil_img, config=config)
            text = text.strip()

            # Add the text to the image
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.4
            font_color = (255, 0, 0)  # Red color for text overlay
            cv2.putText(img_with_cells, f"Cell {i}: {text[:10]}...",
                        (x1, y1 - 5), font, font_scale, font_color, 1)

            # Store cell data with information about merged status
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
                'col': None  # To be determined
            })
            #Debugging print statements.
            print(f"Cell {i}: Width {cell_width}, Height {cell_height}, Merged Horiz: {is_merged_horiz}, Merged Vert: {is_merged_vert}")
            print(f"Cell {i}: Tesseract Output '{text}'")

        except Exception as e:
            print(f"Warning: Error processing cell {i}: {str(e)}")

    return cell_data, img_with_cells

def main():
    pdf_path ="./data/HDFCNew.pdf"
    images_folder ="./data/images"
    image_path = "./data/images/page_1.png"
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
    horizontal_lines = cv2.HoughLines(binary, 1, np.pi / 180, 200)
    vertical_lines = cv2.HoughLines(binary, 1, np.pi / 90, 200)

    # Step 7: Identify cells from the grid, passing the detected lines
    cells = identify_cells_from_grid_neww(intersections, img.shape[:2], h_segments, v_segments)
    # Save the image with cell outlines
    #cv2.imwrite("verified_cells.png", img_with_cells)  
# Step 7: Extract text from cells
    cell_data, img_with_cells = extract_text_from_cells_neww(cells, gray, img)
    cv2.imwrite(os.path.join(output_folder, f"detected_cells_{i+1}.png"), img_with_cells)
    
if __name__ == "__main__":
    main()