import numpy as np
import os
import cv2
from PIL import Image, ImageDraw, ImageFont

def calculate_2d_projections(coordinates_3d, intrinsics):
    """
    Input: 
        coordinates: [3, N]
        intrinsics: [3, 3]
    Return 
        projected_coordinates: [N, 2]
    """
    projected_coordinates = intrinsics @ coordinates_3d
    projected_coordinates = projected_coordinates[:2, :] / projected_coordinates[2, :]
    projected_coordinates = projected_coordinates.transpose()
    projected_coordinates = np.array(projected_coordinates, dtype=np.int32)

    return projected_coordinates

def get_3d_bbox(scale, shift = 0):
    """
    Input: 
        scale: [3] or scalar
        shift: [3] or scalar
    Return 
        bbox_3d: [3, N]

    """
    if hasattr(scale, "__iter__"):
        bbox_3d = np.array([[scale[0] / 2, +scale[1] / 2, scale[2] / 2],
                  [scale[0] / 2, +scale[1] / 2, -scale[2] / 2],
                  [-scale[0] / 2, +scale[1] / 2, scale[2] / 2],
                  [-scale[0] / 2, +scale[1] / 2, -scale[2] / 2],
                  [+scale[0] / 2, -scale[1] / 2, scale[2] / 2],
                  [+scale[0] / 2, -scale[1] / 2, -scale[2] / 2],
                  [-scale[0] / 2, -scale[1] / 2, scale[2] / 2],
                  [-scale[0] / 2, -scale[1] / 2, -scale[2] / 2]]) + shift
    else:
        bbox_3d = np.array([[scale / 2, +scale / 2, scale / 2],
                  [scale / 2, +scale / 2, -scale / 2],
                  [-scale / 2, +scale / 2, scale / 2],
                  [-scale / 2, +scale / 2, -scale / 2],
                  [+scale / 2, -scale / 2, scale / 2],
                  [+scale / 2, -scale / 2, -scale / 2],
                  [-scale / 2, -scale / 2, scale / 2],
                  [-scale / 2, -scale / 2, -scale / 2]]) +shift

    bbox_3d = bbox_3d.transpose()
    return bbox_3d

def draw_3d_bbox(img, imgpts, color, size=3):
    imgpts = np.int32(imgpts).reshape(-1, 2)

    # draw ground layer in darker color
    color_ground = (int(color[0] * 0.3), int(color[1] * 0.3), int(color[2] * 0.3))
    for i, j in zip([4, 5, 6, 7],[5, 7, 4, 6]):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), color_ground, size)

    # draw pillars in blue color
    color_pillar = (int(color[0]*0.6), int(color[1]*0.6), int(color[2]*0.6))
    for i, j in zip(range(4),range(4,8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), color_pillar, size)

    # finally, draw top layer in color
    for i, j in zip([0, 1, 2, 3],[1, 3, 0, 2]):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), color, size)
    return img

def draw_3d_pts(img, imgpts, color, size=1):
    imgpts = np.int32(imgpts).reshape(-1, 2)
    for point in imgpts:
        img = cv2.circle(img, (point[0], point[1]), size, color, -1)
    return img

def draw_detections(image, pred_rots, pred_trans, pose_scores, model_points, intrinsics, color=(255, 0, 0), text_color=(255, 255, 0), font_size=15):
    num_pred_instances = len(pred_rots)
    draw_image_bbox = image.copy()
    # 3d bbox
    scale = (np.max(model_points, axis=0) - np.min(model_points, axis=0))
    shift = np.mean(model_points, axis=0)
    bbox_3d = get_3d_bbox(scale, shift)

    # 3d point
    choose = np.random.choice(np.arange(len(model_points)), 512)
    pts_3d = model_points[choose].T

    all_projected_bboxes = []

    for ind in range(num_pred_instances):
        # draw 3d bounding box
        transformed_bbox_3d = pred_rots[ind]@bbox_3d + pred_trans[ind][:,np.newaxis]
        projected_bbox = calculate_2d_projections(transformed_bbox_3d, intrinsics[ind])
        all_projected_bboxes.append(projected_bbox) # Store for later use
        draw_image_bbox = draw_3d_bbox(draw_image_bbox, projected_bbox, color)
        # draw point cloud
        transformed_pts_3d = pred_rots[ind]@pts_3d + pred_trans[ind][:,np.newaxis]
        projected_pts = calculate_2d_projections(transformed_pts_3d, intrinsics[ind])
        draw_image_bbox = draw_3d_pts(draw_image_bbox, projected_pts, color)

    if pose_scores is None or len(pose_scores) == 0:
        return draw_image_bbox
    
    # 2nd drawing stage: Render the score of each detection on the image

    # --- Now draw all text using PIL ---
    # Convert the image with geometry drawn to PIL format ONCE
    pil_image = Image.fromarray(draw_image_bbox)
    draw = ImageDraw.Draw(pil_image)
    try:
        # Try loading a specific font like DejaVuSans
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except IOError:
        # Fallback to default font if specific font not found
        print("DejaVuSans font not found, using default PIL font.")
        # Default font might not support size well, use default size
        font = ImageFont.load_default()
        # Adjust font size for default font - this might not work as expected
        # font = ImageFont.load_default().font_variant(size=font_size) # More modern approach if needed


    # Iterate again to draw text using stored bbox projections
    img_h, img_w = draw_image_bbox.shape[:2] # Get image dimensions

    for ind in range(num_pred_instances):
        score = pose_scores[ind]
        text = f"Score: {score:.2f}"
        projected_bbox = all_projected_bboxes[ind] # Get the stored projection

        # Calculate position based on projected bbox
        if projected_bbox.size > 0:
            # Filter out potential non-finite values from projection if necessary
            valid_coords = projected_bbox[:, np.all(np.isfinite(projected_bbox), axis=0)]

            if valid_coords.size > 0:
                min_x = np.min(valid_coords[:, 0])
                min_y = np.min(valid_coords[:, 1])

                # Use textbbox in modern Pillow to get text size for better positioning
                try:
                    # left, top, right, bottom
                    bbox = draw.textbbox((0, 0), text, font=font)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]
                except AttributeError:
                    # Fallback for older Pillow versions
                    text_width, text_height = draw.textlength(text, font=font), font_size # Approximate height

                # Position text slightly above the minimum y coordinate of the bbox
                text_x = int(min_x)
                text_y = int(min_y - text_height - 2) # Place text above the box top

                # Clamp coordinates to ensure text starts within image bounds
                # Ensure text doesn't go *off* the top or left edge
                text_x = max(0, text_x)
                text_y = max(0, text_y)
                # Ensure text doesn't go *too far* off the right/bottom (less critical for start pos)
                text_x = min(text_x, img_w - 10) # Leave some margin
                text_y = min(text_y, img_h - 10)

                position = (text_x, text_y)
                # Draw text background rectangle (optional, for visibility)
                # bg_rect = [position[0]-2, position[1]-2, position[0] + text_width + 2, position[1] + text_height + 2]
                # draw.rectangle(bg_rect, fill=(0,0,0,128)) # Semi-transparent black bg

                # Draw the text
                draw.text(position, text, fill=text_color, font=font)
            else:
                print(f"Warning: No valid projected bbox points for instance {ind} to draw score.")
                # Optionally draw score at a default location if bbox invalid/empty
                # draw.text((10, 10 + ind * (font_size+5)), text, fill=text_color, font=font)
        else:
            print(f"Warning: Projected bbox is empty for instance {ind}. Cannot draw score.")
            # Optionally draw score at default location
            # draw.text((10, 10 + ind * (font_size+5)), text, fill=text_color, font=font)


    # Convert the final PIL image (with geometry and text) back to NumPy array
    draw_image_bbox = np.array(pil_image)

    return draw_image_bbox
