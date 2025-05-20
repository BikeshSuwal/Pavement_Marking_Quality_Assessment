import cv2
import numpy as np
from pathlib import Path
import concurrent.futures
from tqdm import tqdm

INPUT_DIR = Path("mapillary_images")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# How many worker threads to use (None for auto-detection)
MAX_WORKERS = None

def apply_CLAHE(image):
    """Apply Contrast Limited Adaptive Histogram Equalization to enhance image contrast"""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    lab_enhanced = cv2.merge((l_enhanced, a, b))
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

def detect_lane_lines(image):
    """Detect lane lines to use for perspective transformation points"""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Define region of interest (ROI) - lower half of the image
    height, width = edges.shape
    mask = np.zeros_like(edges)
    roi_vertices = np.array([
        [0, height], 
        [0, height * 0.6], 
        [width, height * 0.6], 
        [width, height]
    ], np.int32)
    cv2.fillPoly(mask, [roi_vertices], 255)
    masked_edges = cv2.bitwise_and(edges, mask)
    
    # Detect lines using Hough transform
    lines = cv2.HoughLinesP(
        masked_edges, 
        rho=1, 
        theta=np.pi/180, 
        threshold=20, 
        minLineLength=height/4, 
        maxLineGap=height/8
    )
    
    # If no lines detected, return None
    if lines is None:
        return None
    
    # Separate lines into left and right based on slope
    left_lines, right_lines = [], []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # Calculate slope
        if x2 - x1 == 0:  # Vertical line
            continue
        slope = (y2 - y1) / (x2 - x1)
        # Separate lines based on slope
        if slope < -0.3:  # Left lane
            left_lines.append(line[0])
        elif slope > 0.3:  # Right lane
            right_lines.append(line[0])
    
    # If insufficient lines detected, return None
    if len(left_lines) < 2 or len(right_lines) < 2:
        return None
    
    # Create a visual debug image 
    # Uncomment for debugging
    # line_image = np.zeros_like(image)
    # for line in left_lines + right_lines:
    #     x1, y1, x2, y2 = line
    #     cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # cv2.imwrite("debug_lines.jpg", line_image)
    
    return left_lines, right_lines

def find_lane_endpoints(lines, height, is_left=True):
    """Find top and bottom endpoints of lane lines"""
    if not lines:
        return None
    
    # Extract all x-coordinates at two y positions (top and bottom of ROI)
    top_y = int(height * 0.6)
    bottom_y = height - 1
    
    top_points, bottom_points = [], []
    
    for line in lines:
        x1, y1, x2, y2 = line
        
        # Line equation: x = ((y - y1) * (x2 - x1) / (y2 - y1)) + x1
        if y2 - y1 != 0:
            # Calculate x at top_y
            if min(y1, y2) <= top_y <= max(y1, y2):
                x_at_top = int(((top_y - y1) * (x2 - x1) / (y2 - y1)) + x1)
                top_points.append(x_at_top)
            
            # Calculate x at bottom_y
            if min(y1, y2) <= bottom_y <= max(y1, y2):
                x_at_bottom = int(((bottom_y - y1) * (x2 - x1) / (y2 - y1)) + x1)
                bottom_points.append(x_at_bottom)
    
    # If we don't have enough points, return None
    if not top_points or not bottom_points:
        return None
    
    # For left lane, we want the rightmost point at top and bottom
    # For right lane, we want the leftmost point at top and bottom
    if is_left:
        top_x = max(top_points)
        bottom_x = max(bottom_points)
    else:
        top_x = min(top_points)
        bottom_x = min(bottom_points)
    
    return [(top_x, top_y), (bottom_x, bottom_y)]

def adaptive_perspective_warp(image):
    """Apply perspective warping based on detected lane lines"""
    height, width = image.shape[:2]
    
    # Try to detect lane lines
    lane_detection = detect_lane_lines(image)
    
    if lane_detection:
        left_lines, right_lines = lane_detection
        
        # Find endpoints for the lanes
        left_endpoints = find_lane_endpoints(left_lines, height, is_left=True)
        right_endpoints = find_lane_endpoints(right_lines, height, is_left=False)
        
        # If we successfully found all endpoints
        if left_endpoints and right_endpoints:
            # Source points - detected lane line endpoints
            src = np.float32([
                left_endpoints[0],  # Top left
                right_endpoints[0],  # Top right
                right_endpoints[1],  # Bottom right
                left_endpoints[1]   # Bottom left
            ])
            
            # Destination points - rectangular shape
            margin = width * 0.2  # 20% margin from edges
            dst = np.float32([
                [margin, 0],               # Top left
                [width - margin, 0],       # Top right
                [width - margin, height],  # Bottom right
                [margin, height]           # Bottom left
            ])
            
            # Calculate transformation matrix
            M = cv2.getPerspectiveTransform(src, dst)
            warped = cv2.warpPerspective(image, M, (width, height), flags=cv2.INTER_LINEAR)
            return warped
    
    # Fallback to default transformation if lane detection fails
    return default_perspective_warp(image)

def default_perspective_warp(image):
    """Default perspective transformation when lane detection fails"""
    height, width = image.shape[:2]
    
    # Source points - assuming a trapezoidal road area
    src = np.float32([
        [width * 0.4, height * 0.65],   # Top left
        [width * 0.6, height * 0.65],   # Top right
        [width * 0.9, height * 0.95],   # Bottom right
        [width * 0.1, height * 0.95]    # Bottom left
    ])
    
    # Destination points - rectangular shape
    dst = np.float32([
        [width * 0.2, 0],          # Top left
        [width * 0.8, 0],          # Top right
        [width * 0.8, height],     # Bottom right
        [width * 0.2, height]      # Bottom left
    ])
    
    # Calculate transformation matrix
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(image, M, (width, height), flags=cv2.INTER_LINEAR)
    return warped

def preprocess_image(image_path):
    """Process a single image with CLAHE enhancement and perspective correction"""
    try:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Failed to load {image_path}")
            return False
        
        # Apply CLAHE for contrast enhancement
        enhanced = apply_CLAHE(image)
        
        # Apply perspective correction
        corrected = adaptive_perspective_warp(enhanced)
        
        # Save the result
        output_path = OUTPUT_DIR / image_path.name
        cv2.imwrite(str(output_path), corrected)
        return True
    except Exception as e:
        print(f"Error processing {image_path.name}: {str(e)}")
        return False

def main():
    # Get all jpg images
    image_paths = list(INPUT_DIR.glob("*.jpg"))
    if not image_paths:
        print(f"No JPG images found in {INPUT_DIR}")
        return
    
    print(f"Found {len(image_paths)} images to process")
    
    # Determine number of workers
    if MAX_WORKERS is None:
        # Use 75% of available cores by default
        num_workers = max(1, int(min(len(image_paths), (cv2.getNumberOfCPUs() * 0.75))))
    else:
        num_workers = MAX_WORKERS
    
    print(f"Using {num_workers} worker threads")
    
    # Process images in parallel
    success_count = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(executor.map(preprocess_image, image_paths), total=len(image_paths)))
        success_count = sum(1 for r in results if r)
    
    print(f"Successfully processed {success_count} of {len(image_paths)} images")

if __name__ == "__main__":
    main()