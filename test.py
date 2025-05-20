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
    
    # Advanced preprocessing to enhance lane visibility
    # Apply adaptive thresholding to better identify lane markings
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    # Enhance contrast in lower half of the image
    height, width = thresh.shape
    lower_half = thresh[int(height * 0.5):, :]
    lower_half = cv2.equalizeHist(lower_half)
    thresh[int(height * 0.5):, :] = lower_half
    
    # Apply morphological operations to enhance lane lines
    kernel = np.ones((3, 3), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # Edge detection with optimized parameters
    edges = cv2.Canny(morph, 50, 150)
    
    # Define expanded region of interest (ROI) - more of the image
    mask = np.zeros_like(edges)
    roi_vertices = np.array([
        [0, height], 
        [0, height * 0.5],  # Increased from 0.6 to 0.5 to capture more road area
        [width, height * 0.5],  # Increased from 0.6 to 0.5
        [width, height]
    ], np.int32)
    cv2.fillPoly(mask, [roi_vertices], 255)
    masked_edges = cv2.bitwise_and(edges, mask)
    
    # Use color information to enhance lane detection
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Yellow lane mask
    yellow_lower = np.array([15, 80, 120])
    yellow_upper = np.array([35, 255, 255])
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
    # White lane mask
    white_lower = np.array([0, 0, 200])
    white_upper = np.array([180, 30, 255])
    white_mask = cv2.inRange(hsv, white_lower, white_upper)
    # Combine masks and apply to edges
    color_mask = cv2.bitwise_or(yellow_mask, white_mask)
    color_mask = cv2.bitwise_and(color_mask, mask)  # Apply ROI to color mask
    color_edges = cv2.bitwise_and(edges, color_mask)
    
    # Combine color-based and edge-based detection
    combined_edges = cv2.bitwise_or(masked_edges, color_edges)
    
    # Detect lines using Hough transform with optimized parameters
    lines = cv2.HoughLinesP(
        combined_edges, 
        rho=1, 
        theta=np.pi/180, 
        threshold=15,  # Reduced to detect more lines
        minLineLength=height/5,  # Shorter minimum length to detect more lines
        maxLineGap=height/6      # Slightly increased gap tolerance
    )
    
    # If no lines detected, return None
    if lines is None:
        return None
    
    # Separate lines into left and right based on slope and position
    left_lines, right_lines = [], []
    
    # Filter and categorize detected lines
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # Skip too horizontal lines (likely not lane markers)
        if abs(y2 - y1) < height * 0.05:
            continue
            
        # Calculate slope
        if x2 - x1 == 0:  # Vertical line
            continue
        slope = (y2 - y1) / (x2 - x1)
        
        # Line midpoint x-coordinate to help with classification
        midpoint_x = (x1 + x2) / 2
        
        # Better line classification using both slope and position
        if slope < -0.25 and midpoint_x < width * 0.6:  # Left lane
            left_lines.append(line[0])
        elif slope > 0.25 and midpoint_x > width * 0.4:  # Right lane
            right_lines.append(line[0])
    
    # More lenient threshold for line detection success
    if len(left_lines) < 1 or len(right_lines) < 1:
        return None
    
    # Create a visual debug image (uncomment for debugging)
    # debug_image = image.copy()
    # for line in left_lines:
    #     x1, y1, x2, y2 = line
    #     cv2.line(debug_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    # for line in right_lines:
    #     x1, y1, x2, y2 = line
    #     cv2.line(debug_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # cv2.imwrite("debug_lines.jpg", debug_image)
    
    return left_lines, right_lines

def find_lane_endpoints(lines, height, width, is_left=True):
    """Find top and bottom endpoints of lane lines with improved extrapolation"""
    if not lines:
        return None
    
    # Extract all points from the lines
    all_points = []
    for line in lines:
        x1, y1, x2, y2 = line
        all_points.append((x1, y1))
        all_points.append((x2, y2))
    
    # Extract all x-coordinates at multiple y positions
    top_y = int(height * 0.5)  # Changed from 0.6 to 0.5 to capture more road area
    middle_y = int(height * 0.75)
    bottom_y = height - 1
    
    # Fit a polynomial to the points to better handle curves
    x_coords = [p[0] for p in all_points]
    y_coords = [p[1] for p in all_points]
    
    # If we don't have enough points, use simple linear approach
    if len(x_coords) < 3:
        # Use line interpolation if polynomial fit isn't possible
        top_points, bottom_points = [], []
        
        for line in lines:
            x1, y1, x2, y2 = line
            
            # Line equation: x = ((y - y1) * (x2 - x1) / (y2 - y1)) + x1
            if y2 - y1 != 0:
                # Calculate x at top_y (extrapolate if needed)
                x_at_top = int(((top_y - y1) * (x2 - x1) / (y2 - y1)) + x1)
                top_points.append(x_at_top)
                
                # Calculate x at bottom_y
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
            
    else:
        # Use polynomial fit for better curve handling when we have enough points
        try:
            # Use 2nd degree polynomial to handle curved lanes
            z = np.polyfit(y_coords, x_coords, 2)
            lane_fit = np.poly1d(z)
            
            # Calculate x-coordinates at our target y positions
            top_x = int(lane_fit(top_y))
            bottom_x = int(lane_fit(bottom_y))
            
            # Sanity checks and boundary enforcement
            if is_left:
                # Enforce reasonable bounds for left lane
                top_x = max(0, min(top_x, width // 2))
                bottom_x = max(0, min(bottom_x, width // 2))
            else:
                # Enforce reasonable bounds for right lane
                top_x = max(width // 2, min(top_x, width))
                bottom_x = max(width // 2, min(bottom_x, width))
                
        except Exception:
            # Fallback to linear approach if polynomial fit fails
            if is_left:
                top_x = int(width * 0.3)
                bottom_x = int(width * 0.1) 
            else:
                top_x = int(width * 0.7)
                bottom_x = int(width * 0.9)
    
    # Return the endpoints
    return [(top_x, top_y), (bottom_x, bottom_y)]

def adaptive_perspective_warp(image):
    """Apply perspective warping based on detected lane lines with wider coverage"""
    height, width = image.shape[:2]
    
    # Try to detect lane lines
    lane_detection = detect_lane_lines(image)
    
    if lane_detection:
        left_lines, right_lines = lane_detection
        
        # Find endpoints for the lanes with improved extrapolation
        left_endpoints = find_lane_endpoints(left_lines, height, width, is_left=True)
        right_endpoints = find_lane_endpoints(right_lines, height, width, is_left=False)
        
        # If we successfully found all endpoints
        if left_endpoints and right_endpoints:
            # Source points - detected lane line endpoints
            top_left, bottom_left = left_endpoints
            top_right, bottom_right = right_endpoints
            
            # Calculate additional points to extend the warping area
            # This creates a wider trapezoid that captures more road area
            road_width = max(top_right[0] - top_left[0], bottom_right[0] - bottom_left[0])
            
            # Extended source points - add extra width on both sides
            extra_width_factor = 0.5  # Extend by 50% on each side
            
            # Calculate extended points
            ext_top_left = (max(0, int(top_left[0] - road_width * extra_width_factor)), top_left[1])
            ext_top_right = (min(width-1, int(top_right[0] + road_width * extra_width_factor)), top_right[1])
            ext_bottom_left = (max(0, int(bottom_left[0] - road_width * extra_width_factor)), bottom_left[1])
            ext_bottom_right = (min(width-1, int(bottom_right[0] + road_width * extra_width_factor)), bottom_right[1])
            
            src = np.float32([
                ext_top_left,      # Extended top left
                ext_top_right,     # Extended top right
                ext_bottom_right,  # Extended bottom right
                ext_bottom_left    # Extended bottom left
            ])
            
            # Destination points - rectangular shape with wider coverage
            dst_margin = width * 0.05  # Reduced margin for wider coverage (from 0.2 to 0.05)
            dst = np.float32([
                [dst_margin, 0],                # Top left
                [width - dst_margin, 0],        # Top right
                [width - dst_margin, height],   # Bottom right
                [dst_margin, height]            # Bottom left
            ])
            
            # Calculate transformation matrix
            M = cv2.getPerspectiveTransform(src, dst)
            warped = cv2.warpPerspective(image, M, (width, height), flags=cv2.INTER_LINEAR)
            
            # Store the transformation for inverse if needed
            # Minv = cv2.getPerspectiveTransform(dst, src)
            
            return warped
    
    # Fallback to enhanced default transformation if lane detection fails
    return enhanced_default_warp(image)

def enhanced_default_warp(image):
    """Improved default perspective transformation when lane detection fails"""
    height, width = image.shape[:2]
    
    # Source points - using a wider trapezoid with higher vanishing point
    src = np.float32([
        [width * 0.25, height * 0.5],    # Top left (moved up from 0.65 to 0.5)
        [width * 0.75, height * 0.5],    # Top right (moved up from 0.65 to 0.5)
        [width * 0.95, height * 0.95],   # Bottom right (wider from 0.9 to 0.95)
        [width * 0.05, height * 0.95]    # Bottom left (wider from 0.1 to 0.05)
    ])
    
    # Destination points - rectangular shape with minimal margins
    dst = np.float32([
        [width * 0.05, 0],          # Top left (from 0.2 to 0.05)
        [width * 0.95, 0],          # Top right (from 0.8 to 0.95)
        [width * 0.95, height],     # Bottom right (from 0.8 to 0.95)
        [width * 0.05, height]      # Bottom left (from 0.2 to 0.05)
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
        
        # Apply stronger CLAHE for better contrast enhancement
        enhanced = apply_enhanced_CLAHE(image)
        
        # Apply perspective correction
        corrected = adaptive_perspective_warp(enhanced)
        
        # Optionally apply post-processing to improve the top-view quality
        corrected = post_process_warped_image(corrected)
        
        # Save the result
        output_path = OUTPUT_DIR / image_path.name
        cv2.imwrite(str(output_path), corrected)
        
        # Generate a comparison image (side-by-side)
        comparison = create_comparison_image(image, corrected)
        comparison_path = OUTPUT_DIR / f"compare_{image_path.name}"
        cv2.imwrite(str(comparison_path), comparison)
        
        return True
    except Exception as e:
        print(f"Error processing {image_path.name}: {str(e)}")
        return False

def apply_enhanced_CLAHE(image):
    """Apply enhanced Contrast Limited Adaptive Histogram Equalization"""
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel with stronger parameters
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    
    # Enhance shadows in the lower part of the image
    h, w = l_enhanced.shape
    lower_roi = l_enhanced[int(h*0.6):h, :]
    lower_roi = cv2.equalizeHist(lower_roi)
    l_enhanced[int(h*0.6):h, :] = lower_roi
    
    # Merge channels and convert back to BGR
    lab_enhanced = cv2.merge((l_enhanced, a, b))
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

def post_process_warped_image(warped):
    """Apply post-processing to improve warped image quality"""
    # Apply sharpening to enhance details in the warped image
    kernel = np.array([[-1, -1, -1], 
                       [-1,  9, -1], 
                       [-1, -1, -1]])
    sharpened = cv2.filter2D(warped, -1, kernel)
    
    # Blend the sharpened image with the original
    alpha = 0.7
    blended = cv2.addWeighted(warped, alpha, sharpened, 1-alpha, 0)
    
    # Optional: Crop the bottom part to remove any artifacts
    # height = blended.shape[0]
    # blended = blended[:int(height * 0.95), :]
    
    return blended

def create_comparison_image(original, warped):
    """Create a side-by-side comparison of original and warped images"""
    # Resize images to same height if needed
    h1, w1 = original.shape[:2]
    h2, w2 = warped.shape[:2]
    
    # Create target height (max of both images)
    target_h = max(h1, h2)
    
    # Scale widths proportionally to target height
    new_w1 = int(w1 * (target_h / h1))
    new_w2 = int(w2 * (target_h / h2))
    
    # Resize images
    original_resized = cv2.resize(original, (new_w1, target_h))
    warped_resized = cv2.resize(warped, (new_w2, target_h))
    
    # Concatenate horizontally
    comparison = np.concatenate((original_resized, warped_resized), axis=1)
    
    # Add labels
    cv2.putText(comparison, "Original", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(comparison, "Bird's-eye View", (new_w1 + 10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    return comparison

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