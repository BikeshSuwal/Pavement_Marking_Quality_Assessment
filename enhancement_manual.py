import cv2
import numpy as np
import os
import sys
from pathlib import Path


class PerspectiveCorrector:
    def __init__(self, resize_width=800):
        self.resize_width = resize_width
        self.points = []
        self.window_name = "Perspective Correction - Select 4 Points"

    # Add this function to your PerspectiveCorrector class
    def apply_clahe(self, image, clip_limit=2.0, tile_grid_size=(8, 8)):
        """
        Apply CLAHE preprocessing to enhance contrast in road markings.
        
        Parameters:
        - image: Input image (assumed BGR)
        - clip_limit: Threshold for contrast limiting
        - tile_grid_size: Size of grid for histogram equalization
        
        Returns:
        - Enhanced image after CLAHE
        """
        # Convert to LAB colour space for better luminance separation
        lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        # Split LAB image into individual channels
        l_channel, a_channel, b_channel = cv2.split(lab_image)

        # Create CLAHE object and apply it to the L channel
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        l_channel_clahe = clahe.apply(l_channel)

        # Merge the CLAHE-enhanced L channel back with a and b channels
        lab_clahe = cv2.merge((l_channel_clahe, a_channel, b_channel))

        # Convert back to BGR colour space
        enhanced_image = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

        return enhanced_image
 
        
    def select_points(self, image_path):
        """Interactive point selection with visualization and validation"""
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not read image {image_path}")
            return None
            
        self.original_h, self.original_w = img.shape[:2]
        self.aspect_ratio = self.original_h / self.original_w
        self.resized_h = int(self.resize_width * self.aspect_ratio)
        
        self.original_img = img.copy()
        self.display_img = cv2.resize(img, (self.resize_width, self.resized_h))
        self.points = []
        
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
        
        instructions = [
            "Select 4 points in order:",
            "1. Top-left road boundary",
            "2. Top-right road boundary",
            "3. Bottom-right road boundary",
            "4. Bottom-left road boundary",
            "ESC: Cancel/Exit | ENTER: Confirm | S: Skip Image"
        ]
        
        while True:
            display_copy = self.display_img.copy()
            
            # Draw instructions
            for i, text in enumerate(instructions):
                cv2.putText(display_copy, text, (10, 20 + i*20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # Draw selected points
            for i, (x, y) in enumerate(self.points):
                resized_x = int(x * (self.resize_width / self.original_w))
                resized_y = int(y * (self.resized_h / self.original_h))
                cv2.circle(display_copy, (resized_x, resized_y), 5, (0, 0, 255), -1)
                cv2.putText(display_copy, str(i+1), (resized_x+10, resized_y+10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Draw lines between points
            if len(self.points) > 1:
                for i in range(len(self.points)-1):
                    pt1 = (int(self.points[i][0] * (self.resize_width / self.original_w)),
                           int(self.points[i][1] * (self.resized_h / self.original_h)))
                    pt2 = (int(self.points[i+1][0] * (self.resize_width / self.original_w)),
                           int(self.points[i+1][1] * (self.resized_h / self.original_h)))
                    cv2.line(display_copy, pt1, pt2, (0, 255, 0), 1)
                    
                # Connect last point to first if we have all 4 points
                if len(self.points) == 4:
                    pt1 = (int(self.points[3][0] * (self.resize_width / self.original_w)),
                           int(self.points[3][1] * (self.resized_h / self.original_h)))
                    pt2 = (int(self.points[0][0] * (self.resize_width / self.original_w)),
                           int(self.points[0][1] * (self.resized_h / self.original_h)))
                    cv2.line(display_copy, pt1, pt2, (0, 255, 0), 1)
            
            cv2.imshow(self.window_name, display_copy)
            key = cv2.waitKey(1) & 0xFF
            
            if key == 13:  # ENTER key
                if len(self.points) == 4:
                    break
                else:
                    print("Please select exactly 4 points")
            elif key == 27:  # ESC key - exit program
                print("Exiting program")
                cv2.destroyAllWindows()
                sys.exit(0)
            elif key == ord('s') or key == ord('S'):  # S key - skip image
                self.points = []
                cv2.destroyAllWindows()
                return 'skip'
        
        cv2.destroyAllWindows()
        return np.float32(self.points) if len(self.points) == 4 else None
    
    def _mouse_callback(self, event, x, y, flags, params):
        """Handle mouse events for point selection"""
        if event == cv2.EVENT_LBUTTONDOWN and len(self.points) < 4:
            original_x = int(x * (self.original_w / self.resize_width))
            original_y = int(y * (self.original_h / self.resized_h))
            self.points.append([original_x, original_y])
    
    def correct_perspective(self, img, src_pts):
        """Perspective correction with validation"""
        if len(src_pts) != 4:
            raise ValueError("Exactly 4 points required for perspective correction")
            
        h, w = img.shape[:2]
        dst_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        
        # Calculate homography matrix with RANSAC for robustness
        H, status = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        if H is None:
            raise RuntimeError("Homography calculation failed")
            
        return cv2.warpPerspective(img, H, (w, h))
    
    def visualize_correction(self, original_img, corrected_img, max_display_width=1200):
        """Create a comparison visualization with controlled size"""
        # Calculate the target size for display
        h1, w1 = original_img.shape[:2]
        h2, w2 = corrected_img.shape[:2]
        
        # Calculate scaling factor to ensure display isn't too large
        scale_factor = min(1.0, max_display_width / (w1 + w2))
        
        # Apply scaling
        display_h = int(h1 * scale_factor)
        display_w1 = int(w1 * scale_factor)
        display_w2 = int(w2 * scale_factor)
        
        # Resize images for display
        display_original = cv2.resize(original_img, (display_w1, display_h))
        display_corrected = cv2.resize(corrected_img, (display_w2, display_h))
        
        # Create combined image
        combined = np.hstack((display_original, display_corrected))
        
        # Add labels
        cv2.putText(combined, "Original", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(combined, "Corrected", (display_w1 + 10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Add a dividing line between images
        cv2.line(combined, (display_w1, 0), (display_w1, display_h), (255, 255, 255), 2)
        
        return combined


def process_single_image(image_path, output_dir="output", max_display_width=1200, skip_existing=True, save_comparison=True, current_count=None, total_count=None, apply_clahe=True, clahe_clip_limit=2.0, clahe_grid_size=(8, 8)):
    """Process a single image with CLAHE enhancement and perspective correction"""
    corrector = PerspectiveCorrector()
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Get file name and check if already processed
    input_file = Path(image_path)
    base_name = input_file.stem
    output_file = Path(output_dir) / input_file.name
    
    # Skip if already processed
    if skip_existing and output_file.exists():
        print(f"Skipping {image_path} - already processed")
        return False
    
    # Load image
    original_img = cv2.imread(image_path)
    if original_img is None:
        print(f"Error loading image: {image_path}")
        return False
    
    # Apply CLAHE enhancement if requested
    if apply_clahe:
        print("Applying CLAHE enhancement...")
        img = corrector.apply_clahe(original_img, clip_limit=clahe_clip_limit, tile_grid_size=clahe_grid_size)

    else:
        img = original_img.copy()
    
    # Show progress indicator in window title
    progress_info = ""
    if current_count is not None and total_count is not None:
        progress_info = f" [{current_count}/{total_count}]"
    
    # Main processing loop - allows redoing point selection if needed
    while True:
        # Create a window with progress info in title
        window_title = f"Perspective Correction - Select 4 Points{progress_info}"
        corrector.window_name = window_title
        
        # Select points (use enhanced image for point selection)
        print(f"\nProcessing{progress_info}: {image_path}")
            # Process image (CLAHE enhancement happens internally but won't be saved)
        src_pts = corrector.select_points(image_path)
        if src_pts is None:
            return False
        
        if isinstance(src_pts, str) and src_pts == 'skip':
            print(f"Skipping image: {image_path}")
            return 'skip'
        
        if src_pts is None:
            print("Point selection cancelled or failed")
            return False
        
        # Apply correction to the ENHANCED image
        try:
            img_to_correct = cv2.imread(image_path)
            corrected_img = corrector.correct_perspective(img, src_pts)
        except Exception as e:
            print(f"Perspective correction failed: {str(e)}")
            return False
        
        # Create comparison image - compare original vs final corrected
        if save_comparison:
            comparison = corrector.visualize_correction(original_img, corrected_img, max_display_width)
            
            # Display results with options to accept, redo, or quit
            results_window = f"Perspective Correction Results{progress_info}"
            cv2.namedWindow(results_window, cv2.WINDOW_NORMAL)
            
            # Use a wider display that's 1.7 times the point selection window width
            display_height, display_width = comparison.shape[:2]
            scale_factor = corrector.resize_width / corrector.original_w
            target_width = int(display_width * scale_factor * 1.7)
            target_height = int(display_height * scale_factor)
            target_width = max(target_width, 1360)
            target_height = max(target_height, 600)
            cv2.resizeWindow(results_window, target_width, target_height)
            
            # Add instructions to the comparison image
            instruction_text = [
                "ENTER: Accept and Save",
                "R: Redo point selection",
                "ESC: Exit program",
                "S: Skip image"
            ]
            
            instruction_y_start = 70
            for i, text in enumerate(instruction_text):
                cv2.putText(comparison, text, (10, instruction_y_start + i*25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            cv2.imshow(results_window, comparison)
            key = cv2.waitKey(0) & 0xFF
            cv2.destroyAllWindows()
            
            if key == 13:  # ENTER - accept and continue
                break
            elif key == ord('r') or key == ord('R'):  # R key - redo points
                print("Redoing point selection")
                continue
            elif key == 27:  # ESC - exit program
                print("Exiting program")
                sys.exit(0)
            elif key == ord('s') or key == ord('S'):  # S key - skip image
                print(f"Skipping image: {image_path}")
                return 'skip'
        else:
            # If not showing comparison, just accept the results
            break
    
    # Save corrected image with same name as original
    cv2.imwrite(str(output_file), corrected_img)
    print(f"Saved corrected image to {output_file}")
    



def batch_process(input_dir, output_dir="output", max_display_width=1200, skip_existing=True, save_comparison=True,apply_clahe=True, clahe_clip_limit=2.0, clahe_grid_size=(8, 8)):
    """Process all images in a directory"""
    image_extensions = (".jpg", ".jpeg", ".png")
    image_paths = [os.path.join(input_dir, f) 
                  for f in os.listdir(input_dir) 
                  if f.lower().endswith(image_extensions)]
    
    if not image_paths:
        print(f"No images found in {input_dir}")
        return
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(exist_ok=True)
    
    # Count how many images already exist in output directory
    existing_files = set(os.path.basename(f) for f in os.listdir(output_dir) 
                       if os.path.isfile(os.path.join(output_dir, f)) and not f.endswith("_comparison.jpg"))
    
    to_process = []
    skipped = []
    
    for image_path in image_paths:
        if skip_existing and os.path.basename(image_path) in existing_files:
            skipped.append(image_path)
        else:
            to_process.append(image_path)
    
    total_images = len(to_process)
    processed_count = 0
    skipped_count = len(skipped)
    user_skipped_count = 0
    
    print(f"Found {len(image_paths)} images in {input_dir}")
    print(f"- {skipped_count} already processed (will be skipped)")
    print(f"- {total_images} images to process")
    
    for i, image_path in enumerate(to_process, 1):
        current_count = i
        print(f"\nProcessing image {current_count}/{total_images} - {os.path.basename(image_path)}")
        
        # Process the image and check return value
        result = process_single_image(
            image_path, 
            output_dir, 
            max_display_width, 
            skip_existing=False,
            save_comparison=save_comparison,
            current_count=current_count,
            total_count=total_images,
            apply_clahe=True,  # Enable CLAHE by default
            clahe_clip_limit=2.0,  # Default clip limit
            clahe_grid_size=(8, 8)  # Default grid size
        )
        
        # Check result - use string comparison for 'skip' case, not equality check with a string
        if isinstance(result, str) and result == 'skip':
            print(f"Image skipped by user: {os.path.basename(image_path)}")
            user_skipped_count += 1
        elif result is not False and result is not 'skip':
            processed_count += 1
    
    # Summary at the end
    print("\n" + "="*50)
    print("PROCESSING SUMMARY")
    print("="*50)
    print(f"Total images found: {len(image_paths)}")
    print(f"Images automatically skipped (already processed): {skipped_count}")
    print(f"Images manually skipped by user: {user_skipped_count}")
    print(f"Images successfully processed: {processed_count}")
    
    remaining = total_images - processed_count - user_skipped_count
    if remaining > 0:
        print(f"Images remaining to process: {remaining}")
            
    print("="*50)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Perspective Correction Tool")
    parser.add_argument("input", help="Input image or directory")
    parser.add_argument("-o", "--output", default="output", help="Output directory")
    parser.add_argument("-w", "--max-width", type=int, default=1200, 
                        help="Maximum width for display (default: 1200)")
    parser.add_argument("--no-skip", action="store_true", 
                        help="Don't skip images that already exist in output directory")
    parser.add_argument("--no-comparison", action="store_true",
                        help="Don't create and save comparison images")
    parser.add_argument("--no-clahe", action="store_true", 
                        help="Don't apply CLAHE enhancement")
    parser.add_argument("--clahe-clip", type=float, default=2.0,
                        help="CLAHE clip limit (default: 2.0)")
    parser.add_argument("--clahe-grid", type=int, nargs=2, default=[8, 8],
                        help="CLAHE grid size as two numbers (default: 8 8)")    
    args = parser.parse_args()
    
    skip_existing = not args.no_skip
    save_comparison = not args.no_comparison
    
    print("\n" + "="*50)
    print("PERSPECTIVE CORRECTION TOOL")
    print("="*50)
    print(f"Input: {args.input}")
    print(f"Output directory: {args.output}")
    print(f"Skip existing: {'No' if args.no_skip else 'Yes'}")
    print(f"Save comparisons: {'No' if args.no_comparison else 'Yes'}")
    print("="*50)
    print("CONTROLS:")
    print("- ESC: Exit program at any time")
    print("- S: Skip current image")
    print("- R: Redo point selection (after viewing comparison)")
    print("- ENTER: Confirm points / Accept result")
    print("="*50 + "\n")
    
    try:
        if os.path.isfile(args.input):
            process_single_image(
                args.input, args.output, args.max_width, 
                skip_existing=skip_existing, 
                save_comparison=save_comparison,
                current_count=1, total_count=1,
                apply_clahe=not args.no_clahe,
                clahe_clip_limit=args.clahe_clip,
                clahe_grid_size=tuple(args.clahe_grid)
            )

        elif os.path.isdir(args.input):
            batch_process(
                args.input, args.output, args.max_width, 
                skip_existing=skip_existing, 
                save_comparison=save_comparison,
                apply_clahe=not args.no_clahe,
                clahe_clip_limit=args.clahe_clip,
                clahe_grid_size=tuple(args.clahe_grid)
            )
        else:
            print(f"Invalid input path: {args.input}")
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    except Exception as e:
        print(f"\nError: {str(e)}")
        # Add stack trace for better debugging
        import traceback
        traceback.print_exc()
    finally:
        print("\nProcessing complete!")
        cv2.destroyAllWindows()
    
    # Example usage:
    # python enhancement_manual.py mapillary_images -output
    # python test_enhancement_perception.py verified_images -o processed_images --clip-limit 2.0 --tile-size 8