"""
Floor Plan Room Boundary Tracer (Auto-Detect All Rooms)
=======================================================
SINGLE FILE - Fully Automated.

HOW TO USE:
  1. Set IMAGE_PATH to your floor plan image.
  2. Run the script! It will automatically find all rooms.

ALGORITHM:
  1. Identifies walls and empty space.
  2. Thickens walls temporarily to close door gaps.
  3. Finds all separated room regions.
  4. Expands the regions back to perfectly match the original accurate walls.
"""

import os
import json
import cv2
import numpy as np
import random

# ==============================================================
#   SETTINGS  (change these)
# ==============================================================

IMAGE_PATH = r"C:\Users\Roshan\Desktop\BISAG\room\Screenshot 2026-05-19 213221.png"
OUTPUT_DIR = "output"

# ==============================================================
#   TUNING  (adjust if rooms are missed or bleeding)
# ==============================================================

# Threshold for walls (0-255). Pixels darker than this are walls.
# Increase if walls are light gray (e.g., 200). Decrease if dark (e.g., 100).
WALL_THRESHOLD = 150

# How much to thicken walls to seal door gaps (in pixels). 
# If rooms bleed into each other, INCREASE this (try 35, 45, 55).
# If small rooms disappear, DECREASE this (try 25, 15).
WALL_DILATE = 35

# Ignore regions smaller than this area (filters out closets/noise).
MIN_ROOM_AREA = 5000

# Polygon simplification (0.001 = highly detailed, 0.02 = blocky/fewer corners)
POLYGON_SIMPLIFY = 0.005

# Show intermediate steps for debugging
SHOW_DEBUG = False


# ==============================================================
#   STEP 1: Load Image
# ==============================================================

def load_image(path):
    print(f"\n[1] Loading: {path}")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: '{path}'")

    ext = os.path.splitext(path)[1].lower()

    if ext == ".pdf":
        from pdf2image import convert_from_path
        pages = convert_from_path(path, dpi=150, first_page=1, last_page=1)
        img = cv2.cvtColor(np.array(pages[0].convert("RGB")), cv2.COLOR_RGB2BGR)
    else:
        img = cv2.imread(path)
        if img is None:
            raise RuntimeError(f"OpenCV could not read: '{path}'")

    print(f"    Image size: {img.shape[1]} x {img.shape[0]} px")
    return img


# ==============================================================
#   STEP 2: Auto-Detect Rooms & Trace Polygons
# ==============================================================

def find_all_rooms(img):
    print(f"\n[2] Processing floor plan...")
    
    H, W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. Threshold: Dark pixels become walls (0), light pixels become space (255)
    _, binary = cv2.threshold(gray, WALL_THRESHOLD, 255, cv2.THRESH_BINARY)
    
    # Work with walls as white (255) for morphology
    walls_base = cv2.bitwise_not(binary)
    
    # Clean up small noise in walls
    clean_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    walls_clean = cv2.morphologyEx(walls_base, cv2.MORPH_CLOSE, clean_kernel)
    
    # space_base is our accurate ground truth for open space (walls are 0, space is 255)
    space_base = cv2.bitwise_not(walls_clean)
    
    # 2. Dilate walls heavily to close door gaps
    # We use an ellipse kernel so it expands smoothly in all directions
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (WALL_DILATE, WALL_DILATE))
    walls_thick = cv2.dilate(walls_clean, dilate_kernel)
    print(f"    Walls dilated by {WALL_DILATE}px to seal doors.")
    
    # space_shrunk contains the isolated room blobs (since doors are now closed)
    space_shrunk = cv2.bitwise_not(walls_thick)
    
    if SHOW_DEBUG:
        cv2.namedWindow("DEBUG: Shrunk Rooms (Doors Closed)", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("DEBUG: Shrunk Rooms (Doors Closed)", 1000, 700)
        cv2.imshow("DEBUG: Shrunk Rooms (Doors Closed)", space_shrunk)
        print("    [DEBUG] Showing shrunk rooms. Press any key to continue.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # 3. Find contours of these isolated room blobs
    contours, _ = cv2.findContours(space_shrunk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    rooms = []
    MAX_ROOM_AREA = (W * H) * 0.8  # Ignore background outside the house
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_ROOM_AREA or area > MAX_ROOM_AREA:
            continue
            
        # Also skip if the blob touches the very edge of the image (likely the outside world)
        x, y, w, h = cv2.boundingRect(cnt)
        if x <= 2 or y <= 2 or x + w >= W - 3 or y + h >= H - 3:
            continue
            
        # 4. For each valid room blob, recover its ACCURATE shape
        # Draw this specific shrunk blob on a blank mask
        blob_mask = np.zeros((H, W), dtype=np.uint8)
        cv2.drawContours(blob_mask, [cnt], -1, 255, -1)
        
        # Dilate the blob back by the same amount the walls were thickened.
        # This expands the room back to the real walls, and bridges straight across doorways!
        blob_expanded = cv2.dilate(blob_mask, dilate_kernel)
        
        # Intersect with the accurate space to snap exactly to the real walls.
        accurate_room_mask = cv2.bitwise_and(blob_expanded, space_base)
        
        # Find the contour of this accurate room mask
        room_contours, _ = cv2.findContours(accurate_room_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not room_contours:
            continue
            
        # Take the largest piece in case of small disconnected noise
        main_contour = max(room_contours, key=cv2.contourArea)
        
        # Simplify to a polygon
        peri = cv2.arcLength(main_contour, True)
        approx = cv2.approxPolyDP(main_contour, POLYGON_SIMPLIFY * peri, True)
        
        polygon = [(int(p[0][0]), int(p[0][1])) for p in approx]
        rooms.append({
            "polygon": polygon,
            "area": cv2.contourArea(main_contour),
            "center": (x + w//2, y + h//2)  # Rough center for labeling
        })
        
    print(f"    Found {len(rooms)} valid rooms.")
    return rooms


# ==============================================================
#   STEP 3: Draw Overlay
# ==============================================================

def draw_overlay(img, rooms):
    print("\n[3] Drawing overlay...")
    out = img.copy()
    
    # Assign a random distinct color to each room
    np.random.seed(42) # For consistent colors across runs
    
    for i, room in enumerate(rooms):
        color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
        pts = np.array(room["polygon"], dtype=np.int32).reshape((-1, 1, 2))
        
        # Semi-transparent fill
        overlay = out.copy()
        cv2.fillPoly(overlay, [pts], color)
        cv2.addWeighted(overlay, 0.3, out, 0.7, 0, out)
        
        # Solid outline
        cv2.polylines(out, [pts], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)
        
        # Draw corner dots
        for pt in room["polygon"]:
            cv2.circle(out, pt, 4, (0, 0, 255), -1)
            
        # Draw room label
        cx, cy = room["center"]
        label = f"Room {i+1}"
        
        # Add a dark background for text readability
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(out, (cx - tw//2 - 5, cy - th//2 - 5), 
                      (cx + tw//2 + 5, cy + th//2 + 5), (0, 0, 0), -1)
        cv2.putText(out, label, (cx - tw//2, cy + th//2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
    return out


# ==============================================================
#   STEP 4: Save Results
# ==============================================================

def save_results(rooms, overlay_img):
    print(f"\n[4] Saving to '{OUTPUT_DIR}/' ...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    json_path = os.path.join(OUTPUT_DIR, "all_rooms_coords.json")
    
    # Format data for JSON
    data = {"rooms": []}
    for i, room in enumerate(rooms):
        room_data = {
            "id": i + 1,
            "vertex_count": len(room["polygon"]),
            "area": room["area"],
            "polygon": [{"index": j, "x": pt[0], "y": pt[1]} for j, pt in enumerate(room["polygon"])]
        }
        data["rooms"].append(room_data)
        
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"    JSON  -> {json_path}")

    img_path = os.path.join(OUTPUT_DIR, "all_rooms_result.png")
    cv2.imwrite(img_path, overlay_img)
    print(f"    Image -> {img_path}")

    return json_path, img_path


# ==============================================================
#   MAIN
# ==============================================================

def main():
    print("=" * 60)
    print("  Floor Plan Tracer  (Auto-Detect All Rooms, High Accuracy)")
    print("=" * 60)

    try:
        img = load_image(IMAGE_PATH)
    except Exception as e:
        print(f"\n[!] ERROR: {e}")
        return

    rooms = find_all_rooms(img)
    
    if not rooms:
        print("\n[!] No rooms were detected.")
        print("    Try adjusting WALL_THRESHOLD or WALL_DILATE.")
        return

    overlay = draw_overlay(img, rooms)
    json_path, img_path = save_results(rooms, overlay)

    print("\n[5] Result window (press any key to close) ...")
    cv2.namedWindow("Detected Rooms", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Detected Rooms", 1200, 850)
    cv2.imshow("Detected Rooms", overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("\n" + "=" * 60)
    print("  DONE")
    print("=" * 60)
    print(f"  Rooms Found : {len(rooms)}")
    print(f"  JSON saved  : {json_path}")
    print(f"  Image saved : {img_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
