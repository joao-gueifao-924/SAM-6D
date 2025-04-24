import os, sys, gc
import numpy as np
import shutil
from tqdm import tqdm
import time
import torch
from PIL import Image, ImageDraw, ImageFont
import logging
import os, sys
import os.path as osp
from hydra import initialize, compose
# set level logging
logging.basicConfig(level=logging.INFO)
import logging
import trimesh
import numpy as np
from hydra.utils import instantiate
import argparse
import glob
from omegaconf import DictConfig, OmegaConf
from torchvision.utils import save_image
import torchvision.transforms as T
import cv2
import imageio
import distinctipy
from skimage.feature import canny
from skimage.morphology import binary_dilation
from segment_anything.utils.amg import rle_to_mask

from utils.poses.pose_utils import get_obj_poses_from_template_level, load_index_level_in_level2
from utils.bbox_utils import CropResizePad
from model.utils import Detections, convert_npz_to_json
from model.loss import Similarity
from utils.inout import load_json, save_json_bop23

# TODO There is a dubplicate implementation of this method in Pose_Estimation_Model/run_inference_custom.py. Unify them.
def str2bool(v):

    print("type(v): ", type(v))
    if isinstance(v, bool):
       return v

    if not isinstance(v, str):
        raise argparse.ArgumentTypeError(f"Expected a string for boolean conversion, got {type(v)}.")

    v = v.lower().strip()
    if v in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        # Raise an error if the input is not a recognizable boolean string
        raise argparse.ArgumentTypeError(f"Boolean value expected (e.g., 'true', 'false', '1', '0'), but received '{v}'.")

inv_rgb_transform = T.Compose(
        [
            T.Normalize(
                mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
            ),
        ]
    )

def visualize(rgb, detections, only_best_detection=True, save_path="tmp.png"):
    img = rgb.copy()
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    colors = distinctipy.get_colors(len(detections))
    alpha = 0.33

    if only_best_detection:
        best_score = 0.
        for mask_idx, det in enumerate(detections):
            if best_score < det['score']:
                best_score = det['score']
                best_det = detections[mask_idx]
        detections = [best_det]

    for mask_idx, det in enumerate(detections):
        mask = rle_to_mask(det["segmentation"])
        edge = canny(mask)
        edge = binary_dilation(edge, np.ones((2, 2)))
        obj_id = det["category_id"]
        temp_id = obj_id - 1

        r = int(255*colors[temp_id][0])
        g = int(255*colors[temp_id][1])
        b = int(255*colors[temp_id][2])
        img[mask, 0] = alpha*r + (1 - alpha)*img[mask, 0]
        img[mask, 1] = alpha*g + (1 - alpha)*img[mask, 1]
        img[mask, 2] = alpha*b + (1 - alpha)*img[mask, 2]   
        img[edge, :] = 255
        best_score < det['score']
        
    
    img = Image.fromarray(np.uint8(img))
    img.save(save_path)
    prediction = Image.open(save_path)
    
    # concat side by side in PIL
    img = np.array(img)
    concat = Image.new('RGB', (img.shape[1] + prediction.size[0], img.shape[0]))
    concat.paste(rgb, (0, 0))
    concat.paste(prediction, (img.shape[1], 0))
    return concat

def visualize_with_score(rgb: Image.Image, detections, only_best_detection=True, save_path="tmp.png"):
    """
    Visualizes segmentations on an image, highlighting masks and rendering scores.

    Args:
        rgb (PIL.Image.Image): The input RGB image.
        detections (list): A list of detection dictionaries, each containing
                           'segmentation' (RLE), 'score' (float), 'category_id' (int).
        only_best_detection (bool): If True, only visualize the detection with the highest score.
        save_path (str): Path to save the intermediate prediction image.

    Returns:
        PIL.Image.Image: An image with the original RGB on the left and the
                         visualized prediction (grayscale background, highlights, scores) on the right.
    """
    # 1. Prepare grayscale background image (PIL)
    # Create a working copy to avoid modifying the original input PIL Image
    pil_img_for_masks = rgb.copy().convert('L').convert('RGB')

    # 2. Convert PIL image to NumPy array for pixel manipulation
    # We need the shape for the mask decoder and NumPy for blending/edges
    img_np = np.array(pil_img_for_masks)
    img_shape = img_np.shape # (height, width, channels)

    # 3. Handle detections and colors
    if not detections:
        print("Warning: No detections provided.")
        # Return early or handle as needed, e.g., concatenate original with itself or blank
        concat = Image.new('RGB', (rgb.width * 2, rgb.height))
        concat.paste(rgb, (0, 0))
        concat.paste(pil_img_for_masks, (rgb.width, 0)) # Paste grayscale version
        return concat

    if only_best_detection:
        best_det = max(detections, key=lambda det: det['score'])
        detections = [best_det]

    # Use fallback color generator if distinctipy is unavailable or causes issues
    colors = distinctipy.get_colors(len(detections))
    #colors = get_distinct_colors(len(detections))
    alpha = 0.33

    # 4. Prepare for drawing text
    # We'll store text details and draw them *after* all masks/edges are applied
    text_to_draw = []
    font_size = 18 # Adjust as needed
    try:
        # Attempt to load a commonly available TrueType font
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        # Fallback to default bitmap font if arial.ttf is not found
        font = ImageFont.load_default()
        font_size = 10 # Default font size is usually smaller
        print("Arial font not found, using default PIL font.")

    # 5. Process each detection: apply mask, edge, store text info
    for mask_idx, det in enumerate(detections):
        # Decode mask - ensuring it has the correct dimensions
        mask = rle_to_mask(det.get("segmentation")) # Use .get for safety

        # Calculate edge (needs boolean mask)
        edge = canny(mask)
        edge = binary_dilation(edge, np.ones((2, 2))) # Adjust dilation kernel if needed

        # Get color components (scaled 0-255)
        # Use mask_idx if only_best_detection=True, otherwise use category_id logic?
        # Assuming colors list matches the filtered detections list index
        temp_id = mask_idx
        r = int(255 * colors[temp_id][0])
        g = int(255 * colors[temp_id][1])
        b = int(255 * colors[temp_id][2])

        # Apply colored mask using alpha blending directly on the NumPy array
        img_np[mask, 0] = alpha * r + (1 - alpha) * img_np[mask, 0]
        img_np[mask, 1] = alpha * g + (1 - alpha) * img_np[mask, 1]
        img_np[mask, 2] = alpha * b + (1 - alpha) * img_np[mask, 2]

        # Apply white edge directly on the NumPy array
        img_np[edge, :] = 255

        # --- Store Text Information ---
        score = det['score']
        text = f"Score: {score:.2f}"

        # Find a position for the text (e.g., top-left corner of mask bounding box)
        y_indices, x_indices = np.where(mask)
        if len(x_indices) > 0 and len(y_indices) > 0:
            min_y, min_x = np.min(y_indices), np.min(x_indices)
            text_x = min_x
            # Place text slightly above the mask's top edge
            text_y = min_y - font_size - 2 # Add a small buffer
            # Clamp coordinates to be within image bounds
            text_x = max(0, min(text_x, img_shape[1] - 10)) # Prevent going off right edge
            text_y = max(0, min(text_y, img_shape[0] - 10)) # Prevent going off bottom edge
        else:
            # Default position if mask is empty or not found (e.g., top-left corner)
            text_x, text_y = 10, 10 + mask_idx * (font_size + 5) # Stagger if multiple fallbacks

        # Store details: position, text, color (e.g., yellow), font
        text_to_draw.append(((text_x, text_y), text, (255, 255, 0), font))
        # --- End Store Text Information ---

    # 6. Convert the final NumPy array back to a PIL Image
    prediction_pil = Image.fromarray(img_np)

    # 7. Draw all the stored text items onto the prediction PIL image
    draw = ImageDraw.Draw(prediction_pil)
    for position, text, color, text_font in text_to_draw:
        # Draw a small black rectangle behind the text for better visibility
        bbox = text_font.getbbox(text)
        # bbox is (left, top, right, bottom)
        # Width = right - left
        text_width = bbox[2] - bbox[0]
        # Height = bottom - top (pixel height of the ink)
        text_height = bbox[3] - bbox[1]

        bg_coords = [
            (position[0] - 1, position[1] - 1),
            (position[0] + text_width + 1, position[1] + text_height + 1)
        ]
        draw.rectangle(bg_coords, fill=(0, 0, 0)) # Black background rectangle
        draw.text(position, text, fill=color, font=text_font)

    # 8. Save the prediction image (optional, as per original code)
    prediction_pil.save(save_path)
    # prediction = Image.open(save_path) # No need to reload, we have prediction_pil

    # 9. Concatenate original RGB and final prediction side-by-side
    concat = Image.new('RGB', (rgb.width + prediction_pil.width, rgb.height))
    concat.paste(rgb, (0, 0)) # Paste original PIL image
    concat.paste(prediction_pil, (rgb.width, 0)) # Paste final prediction PIL image

    return concat


def batch_input_data(depth_path, cam_path, device):
    batch = {}
    cam_info = load_json(cam_path)
    depth = np.array(imageio.imread(depth_path)).astype(np.int32)
    cam_K = np.array(cam_info['cam_K']).reshape((3, 3))
    depth_scale = np.array(cam_info['depth_scale'])

    batch["depth"] = torch.from_numpy(depth).unsqueeze(0).to(device)
    batch["cam_intrinsic"] = torch.from_numpy(cam_K).unsqueeze(0).to(device)
    batch['depth_scale'] = torch.from_numpy(depth_scale).unsqueeze(0).to(device)
    return batch


def infer_on_image(rgb_image, model):
    # run inference
    start_time = time.time()
    detections = model.segmentor_model.generate_masks(rgb_image)
    detections = Detections(detections)
    logging.info(f"detections inference time: {time.time() - start_time:0.3f} seconds")

    start_time = time.time()
    query_decriptors, query_appe_descriptors = model.descriptor_model.forward(rgb_image, detections)
    logging.info(f"descriptors inference time: {time.time() - start_time:0.3f} seconds")

    return detections, query_decriptors, query_appe_descriptors


def run_inference(  model,
                    device,
                    low_gpu_mem_mode,
                    output_dir,
                    cad_model,
                    rgb_image,
                    all_image_detections,
                    query_decriptors,
                    query_appe_descriptors,
                    depth_image,
                    cam_K,
                    depth_scale,
                    min_detection_final_score
                    ):
    start_time = time.time()
    # matching descriptors
    (
        idx_selected_proposals,
        pred_idx_objects,
        semantic_score,
        best_template,
    ) = model.compute_semantic_score(query_decriptors)
    print(f"matching descriptors time: {time.time() - start_time:0.3f} seconds")

    start_time = time.time()

    # update detections
    # Make a shallow copy so that we don't drop proposals from all_image_detections
    # that don't match with the given set of descriptors
    detections = all_image_detections.shallow_copy()
    detections.filter(idx_selected_proposals)
    query_appe_descriptors = query_appe_descriptors[idx_selected_proposals, :]

    # compute the appearance score
    appe_scores, ref_aux_descriptor= model.compute_appearance_score(best_template, pred_idx_objects, query_appe_descriptors)
    

    # compute the geometric score
    batch = {
        "depth": torch.from_numpy(depth_image).unsqueeze(0).to(device),
        "cam_intrinsic": torch.from_numpy(cam_K).unsqueeze(0).to(device),
        "depth_scale": torch.from_numpy(np.array(depth_scale)).unsqueeze(0).to(device),
    }
    template_poses = get_obj_poses_from_template_level(level=2, pose_distribution="all")
    template_poses[:, :3, 3] *= 0.4
    poses = torch.tensor(template_poses).to(torch.float32).to(device)
    model.ref_data["poses"] =  poses[load_index_level_in_level2(0, "all"), :, :]

    model_points = cad_model.sample(2048).astype(np.float32) # / 1000.0 cad model is already provided in meters by IpdReader
    model.ref_data["pointcloud"] = torch.tensor(model_points).unsqueeze(0).data.to(device)
    
    if low_gpu_mem_mode:
        gc.collect()
        torch.cuda.empty_cache()

    image_uv = model.project_template_to_image(best_template, pred_idx_objects, batch, detections.masks)

    geometric_score, visible_ratio = model.compute_geometric_score(
        image_uv, detections, query_appe_descriptors, ref_aux_descriptor, visible_thred=model.visible_thred
        )

    # final score
    final_score = (semantic_score + appe_scores + geometric_score*visible_ratio) / (1 + 1 + visible_ratio)

    print(f"rest of inference time: {time.time() - start_time:0.3f} seconds")

    detections.add_attribute("scores", final_score)
    detections.add_attribute("object_ids", torch.zeros_like(final_score))

    # Discard all detections with low score:
    mask = detections.scores >= min_detection_final_score
    indices = torch.nonzero(mask).flatten()
    detections.filter(indices)

    return detections


def save_output(output_dir, rgb_image, detections, only_best_detection=True):
    detections = detections.to_numpy(inplace=False)
    save_path = f"{output_dir}/sam6d_results/detection_ism"
    os.makedirs(save_path, exist_ok=True)
    detections.save_to_file(0, 0, 0, save_path, "Custom", return_results=False)
    detections = convert_npz_to_json(idx=0, list_npz_paths=[save_path+".npz"])
    save_json_bop23(save_path+".json", detections)
    vis_img = visualize_with_score(Image.fromarray(rgb_image).convert("RGB"), detections, only_best_detection, f"{output_dir}/sam6d_results/vis_ism.png")
    vis_img.save(f"{output_dir}/sam6d_results/vis_ism.png")

def init_templates(output_dir, model, device):
    logging.info("Initializing templates")

    template_dir = os.path.join(output_dir, 'templates')
    descriptors_path = os.path.join(output_dir, "descriptors.pt")
    appe_descriptors_path = os.path.join(output_dir, "appe_descriptors.pt")

    if os.path.exists(descriptors_path) and os.path.exists(appe_descriptors_path):
        logging.info("Loading descriptors from disk.")
        descriptors = torch.load(descriptors_path, map_location=device)
        appe_descriptors = torch.load(appe_descriptors_path, map_location=device)
        return descriptors, appe_descriptors

    num_templates = len(glob.glob(f"{template_dir}/*.npy"))
    boxes, masks, templates = [], [], []
    for idx in range(num_templates):
        image = Image.open(os.path.join(template_dir, 'rgb_'+str(idx)+'.png'))
        mask = Image.open(os.path.join(template_dir, 'mask_'+str(idx)+'.png'))
        boxes.append(mask.getbbox())

        image = torch.from_numpy(np.array(image.convert("RGB")) / 255).float()
        mask = torch.from_numpy(np.array(mask.convert("L")) / 255).float()
        image = image * mask[:, :, None]
        templates.append(image)
        masks.append(mask.unsqueeze(-1))
        
    templates = torch.stack(templates).permute(0, 3, 1, 2)
    masks = torch.stack(masks).permute(0, 3, 1, 2)
    boxes = torch.tensor(np.array(boxes))
    
    processing_config = OmegaConf.create(
        {
            "image_size": 224,
        }
    )
    proposal_processor = CropResizePad(processing_config.image_size)
    templates = proposal_processor(images=templates, boxes=boxes).to(device)
    masks_cropped = proposal_processor(images=masks, boxes=boxes).to(device)

    descriptors = model.descriptor_model.compute_features(
                    templates, token_name="x_norm_clstoken"
                ).unsqueeze(0).data
    appe_descriptors = model.descriptor_model.compute_masked_patch_feature(
                    templates, masks_cropped[:, 0, :, :]
                ).unsqueeze(0).data
    
    torch.save(descriptors, descriptors_path)
    torch.save(appe_descriptors, appe_descriptors_path)
    return descriptors, appe_descriptors


def reset_ref_data(model, descriptors, appe_descriptors):
    model.ref_data = {}
    model.ref_data["descriptors"] = descriptors
    model.ref_data["appe_descriptors"] = appe_descriptors


def load_model(segmentor_model, stability_score_thresh):
    start_time = time.time()

    with initialize(version_base=None, config_path="configs"):
        cfg = compose(config_name='run_inference.yaml')

    if segmentor_model == "sam":
        with initialize(version_base=None, config_path="configs/model"):
            cfg.model = compose(config_name='ISM_sam.yaml')
        cfg.model.segmentor_model.stability_score_thresh = stability_score_thresh
    elif segmentor_model == "fastsam":
        with initialize(version_base=None, config_path="configs/model"):
            cfg.model = compose(config_name='ISM_fastsam.yaml')
    else:
        raise ValueError("The segmentor_model {} is not supported now!".format(segmentor_model))


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # disable struct mode for this node so that we can change it:
    target_node = cfg.model.segmentor_model
    OmegaConf.set_struct(target_node, False)
    target_node.device = str(device)

    logging.info("Initializing model")
    elapsed_time = time.time() - start_time
    logging.info(f"config time: {elapsed_time:0.3f} seconds")

    start_time = time.time()
    model = instantiate(cfg.model)
    elapsed_time = time.time() - start_time
    logging.info(f"model instantiation time: {elapsed_time:0.3f} seconds")

    start_time = time.time()
    
    
    model.descriptor_model.model = model.descriptor_model.model.to(device)
    model.descriptor_model.model.device = device
    # if there is predictor in the model, move it to device
    logging.info(f"Moving models to {device} done!")
    
    elapsed_time = time.time() - start_time
    logging.info(f"model setup time: {elapsed_time:0.3f} seconds")
    return model, device


def load_descriptormodel_to_gpu(model):
    cuda_device = torch.device("cuda")
    model.descriptor_model.model = model.descriptor_model.model.to(cuda_device )
    model.descriptor_model.model.device = cuda_device
    

def unload_descriptormodel_to_cpu(model):
    cpu_device = torch.device("cpu")
    model.descriptor_model.model = model.descriptor_model.model.to(cpu_device)
    model.descriptor_model.model.device = cpu_device


def load_and_run_inference(segmentor_model, output_dir, cad_path, rgb_path, depth_path, cam_path, stability_score_thresh, low_gpu_memory_mode):
    """
    Load the CAD model, RGB image, depth image, and camera information, 
    and call the run_inference function with the loaded data.
    """
    # Load CAD model
    cad_model = trimesh.load(cad_path)

    # Load RGB image as a NumPy array
    rgb_image = np.array(Image.open(rgb_path).convert("RGB"))

    # Load depth image as a NumPy array
    depth_image = np.array(imageio.imread(depth_path)).astype(np.int32)

    # Load camera information
    cam_info = load_json(cam_path)

    cam_K = np.array(cam_info['cam_K']).reshape((3, 3))
    depth_scale = cam_info['depth_scale']

    model, device = load_model(segmentor_model, stability_score_thresh)
    init_templates(output_dir, model, device)

    all_image_detections, query_decriptors, query_appe_descriptors = infer_on_image(rgb_image, model)

    obj_class_detections = run_inference(
        model,
        device,
        low_gpu_memory_mode,
        output_dir,
        cad_model,
        rgb_image,
        all_image_detections,
        query_decriptors,
        query_appe_descriptors,
        depth_image,
        cam_K,
        depth_scale,
        min_detection_final_score=0.0
    )

    save_output(output_dir, rgb_image, obj_class_detections, only_best_detection=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--segmentor_model", default='sam', help="The segmentor model in ISM")
    parser.add_argument("--output_dir", nargs="?", help="Path to root directory of the output")
    parser.add_argument("--cad_path", nargs="?", help="Path to CAD(mm)")
    parser.add_argument("--rgb_path", nargs="?", help="Path to RGB image")
    parser.add_argument("--depth_path", nargs="?", help="Path to Depth image(mm)")
    parser.add_argument("--cam_path", nargs="?", help="Path to camera information")
    parser.add_argument("--stability_score_thresh", default=0.97, type=float, help="stability_score_thresh of SAM")
    parser.add_argument("--low_gpu_memory_mode", type=str2bool, nargs='?', const=True, default=False, metavar="BOOLEAN", help="Use less GPU memory. Allows inference on a GPU with at least 8 GB of RAM")
    args = parser.parse_args()
    os.makedirs(f"{args.output_dir}/sam6d_results", exist_ok=True)

    DO_DEBUG_SESSION = False

    if DO_DEBUG_SESSION:  # Hijack the script arguments
        ROOT_DIR = "/home/joao/source/SAM-6D/SAM-6D"
        CAD_PATH = f"{ROOT_DIR}/Data/Example/obj_000005.ply"    # path to a given cad model(mm)
        RGB_PATH = f"{ROOT_DIR}/Data/Example/rgb.png"           # path to a given RGB image
        DEPTH_PATH = f"{ROOT_DIR}/Data/Example/depth.png"       # path to a given depth map(mm)
        CAMERA_PATH = f"{ROOT_DIR}/Data/Example/camera.json"    # path to given camera intrinsics
        #OUTPUT_DIR = f"{ROOT_DIR}/Data/Example/outputs"         # path to a pre-defined file for saving results
        OUTPUT_DIR = "/home/joao/Downloads/algorithm_output"
        args.segmentor_model = "fastsam"
        args.output_dir = OUTPUT_DIR
        args.cad_path = CAD_PATH
        args.rgb_path = RGB_PATH
        args.depth_path = DEPTH_PATH
        args.cam_path = CAMERA_PATH
        os.chdir(f"{ROOT_DIR}/Instance_Segmentation_Model")

    start_time = time.time()
    load_and_run_inference(
        segmentor_model=args.segmentor_model,
        output_dir=args.output_dir,
        cad_path=args.cad_path,
        rgb_path=args.rgb_path,
        depth_path=args.depth_path,
        cam_path=args.cam_path,
        stability_score_thresh=args.stability_score_thresh,
        low_gpu_memory_mode=args.low_gpu_memory_mode
        
    )
    elapsed_time = time.time() - start_time
    print(f"load_and_run_inference() time: {elapsed_time:0.3f} seconds")
    