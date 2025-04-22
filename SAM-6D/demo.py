import os, sys

# Get the directory where demo.py is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the absolute path to the 'Instance_Segmentation_Model' directory
# This directory contains the 'utils' folder that submodules are trying to import directly.
ism_package_dir = os.path.join(script_dir, 'Instance_Segmentation_Model')

# Add the 'Instance_Segmentation_Model' directory to sys.path
# insert(0, ...) gives it high priority, making Python look here first.
if ism_package_dir not in sys.path:
    sys.path.insert(0, ism_package_dir)



import torch, gc, glob
import numpy as np
from datetime import datetime
import time, logging
import ipdreader
import Instance_Segmentation_Model.run_inference_custom as ISM
import Pose_Estimation_Model.run_inference_custom as PEM
import subprocess


class running_average_calc:
    def __init__(self,):
        self.count = 0.0
        self.avg = 0.0
    def update(self, new_val: float):
        self.count += 1
        self.avg += (new_val - self.avg) / self.count


# TODO: need to wrap __main__ script into a new callable function "run_inference" so that we can import it here:
#from Pose_Estimation_Model.run_inference_custom import run_inference as estimate_poses

logging.basicConfig(level=logging.INFO)
TARGET_OBJECT_IDS = []
TARGET_GROUP_IDS = []
IPD_DATASET_ROOT_DIR = "/ipd" # keep in sync with run_container.sh

DEFAULT_BLENDER_PATH = "/home/joao/Downloads/blender-4.2.1-linux-x64"
DEFAULT_OUTPUT_DIR = "/home/joao/Downloads/algorithm_output"
DEFAULT_OBJECT_MESH_DIR = "/media/joao/061A31701A315E3D2/ipd-dataset/bpc_baseline/datasets/models"


#overwrite the value above for now... #TODO FIX THIS
#IPD_DATASET_ROOT_DIR = "/media/joao/061A31701A315E3D2/ipd-dataset/bpc_baseline/datasets"

OUTPUT_DIR = os.getenv("OUTPUT_DIR", DEFAULT_OUTPUT_DIR)
TEMPLATE_OUTPUT_ROOT_DIR = OUTPUT_DIR + "/sam6d_obj_templates"

OBJECT_MESH_DIR = os.getenv("OBJECT_MESH_DIR", DEFAULT_OBJECT_MESH_DIR)

timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
ALGORITHM_OUTPUT = OUTPUT_DIR + "/" + timestamp

LOW_GPU_MEMORY_MODE = True
SHORTER_SIDE = 720

MIN_SEGMENTATION_FINAL_SCORE= 0.0


# Example of VS Code configuration (launch.json) for debugpy
# in order to attach to running Docker container
# on localhost network:
# 
#   {
#     "version": "0.2.0",
#     "configurations": [
#       {
#         "name": "Python: Remote Attach",
#         "type": "debugpy",
#         "request": "attach",
#         "connect": {
#           "host": "localhost",
#           "port": 5678
#         },
#         "pathMappings": [
#           {
#             "localRoot": "${workspaceFolder}",
#             "remoteRoot": "."
#           }
#         ]
#       }
#     ]
#   }
DEBUG_STEP_THROUGH = False
if DEBUG_STEP_THROUGH:
    import debugpy
    debugpy.listen(("0.0.0.0", 5678))



def render_object_templates(object_class_id, object_meshes_dir, render_dir, template_output_root_dir):
    blender_path = os.getenv('BLENDER_PATH', DEFAULT_BLENDER_PATH)
    TOTAL_TEMPLATES_EXPECTED = 42
    
    cad_path = os.path.join(object_meshes_dir, f"obj_{object_class_id:06d}.ply")

    blenderproc_script_path = os.path.join(render_dir, 'render_custom_templates.py')
    # Make sure it's truly absolute
    blenderproc_script_path = os.path.abspath(blenderproc_script_path)

    this_object_template_output_dir = get_obj_template_dir(object_class_id, template_output_root_dir)

    mask_files = sorted(glob.glob(f'{this_object_template_output_dir}/templates/mask_*.png'))
    rgb_files  = sorted(glob.glob(f'{this_object_template_output_dir}/templates/rgb_*.png'))
    xyz_files  = sorted(glob.glob(f'{this_object_template_output_dir}/templates/xyz_*.npy'))

    # If the images are already there, re-use them. Avoid re-rendering the same templates.
    if (len(mask_files) == len(rgb_files) == len(xyz_files) == TOTAL_TEMPLATES_EXPECTED):
        return

    # --- Construct the command as a list ---
    # This avoids shell injection issues compared to shell=True
    command = [
        'blenderproc',
        'run',
        blenderproc_script_path,
        '--custom-blender-path', blender_path,
        '--output_dir', this_object_template_output_dir,
        '--cad_path', cad_path,
        # '--colorize', 'True' # Add this back if needed, ensure it's part of the list
    ]

    logging.info(f"Executing command: {' '.join(command)}")
    logging.info(f"Working directory: {render_dir}")


    try:
        result = subprocess.run(
            command,
            cwd=render_dir,
            check=True,
            capture_output=True, # Optional: remove if you want output directly in terminal
            text=True           # Optional: remove if you don't need decoded text
        )
        # Optional: Print captured output
        logging.info("Command executed successfully.")
        # print("STDOUT:")
        # print(result.stdout)
        if len(result.stderr) > 0:
            logging.error("STDERR from BlenderProc call: " + result.stderr)

    except subprocess.CalledProcessError as e:
        logging.error(f"Error executing command: {e}")
        logging.error(f"Return code: {e.returncode}")
        # --- Add these lines ---
        logging.error("Captured STDOUT:")
        logging.error(e.stdout) # stdout is captured if capture_output=True
        logging.error("Captured STDERR:") # <--- THIS IS OFTEN MOST IMPORTANT
        logging.error(e.stderr) # stderr is captured if capture_output=True
        # --- End added lines ---
        sys.exit(1)

def get_obj_template_dir(object_class_id, template_output_root_dir):
    this_object_template_output_dir = template_output_root_dir + f"/obj_{object_class_id:06d}"
    return this_object_template_output_dir


if __name__=='__main__':
    if DEBUG_STEP_THROUGH:
        print("Waiting for client to attach...")
        debugpy.wait_for_client()
        print("Client attached. Continuing...")

    reader = ipdreader.IpdReader(root_folder=IPD_DATASET_ROOT_DIR, shorter_side=SHORTER_SIDE)

    camera_id = 1

    start_time_overrall = time.time()
    elapsed_time_inference = 0
    total_inferences_made = 0

    avg_total_inference_time = running_average_calc() # running average of inference time for each image as a whole

    # Get the directory where the current script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
        
    # Define the path to the 'Render' directory relative to the script
    render_dir = os.path.join(script_dir, 'Render')

    original_cwd = os.getcwd()
    target_cwd = os.path.join(script_dir, 'Instance_Segmentation_Model'); os.chdir(target_cwd)
    segmentator_model, device = ISM.load_model(segmentor_model="fastsam", stability_score_thresh=0.97) 
    os.chdir(original_cwd)

    target_cwd = os.path.join(script_dir, 'Pose_Estimation_Model'); os.chdir(target_cwd)
    pem = PEM.PoseEstimatorModel()
    os.chdir(original_cwd)

    if LOW_GPU_MEMORY_MODE:
        ISM.load_model_to_gpu(segmentator_model)
        pem.unload_model_to_cpu()
        gc.collect()
        torch.cuda.empty_cache()

    for group_id in reader.enumerate_groups():
        if TARGET_GROUP_IDS is not None and len(TARGET_GROUP_IDS) > 0 and object_class_id not in TARGET_GROUP_IDS:
            continue
        for scene_id in reader.enumerate_scenes(group_id):
            logging.info(f"Start of scene: {scene_id}")
            color = reader.get_rgb_image(group_id, scene_id, camera_id)
            depth = reader.get_depth_image(group_id, scene_id, camera_id)
            K = reader.get_camera_intrinsics_K_matrix(group_id, scene_id, camera_id)

            whole_pts = pem.get_point_cloud_from_depth(depth, K)
            
            all_image_detections, query_decriptors, query_appe_descriptors = ISM.infer_on_image(color, segmentator_model)

            total_known_objs_present = len(reader.enumerate_objects(group_id, scene_id, camera_id).items())
            logging.info(f"Total of {total_known_objs_present} known objects are present in group {group_id}, scene {scene_id}, cam {camera_id}")

            for object_class_id, object_instance_ids in reader.enumerate_objects(group_id, scene_id, camera_id).items():
                if TARGET_OBJECT_IDS is not None and len(TARGET_OBJECT_IDS) > 0 and object_class_id not in TARGET_OBJECT_IDS:
                    continue
                
                start_whole_image_inference_time = time.time()

                # render_object_templates is smart in the sense that 
                # it avoid re-rendering templates for the same object class if its
                # corresponding  template images are already present in its cache folder.
                start_time = time.time()
                mesh = reader.get_object_mesh(object_class_id)
                render_object_templates(object_class_id, OBJECT_MESH_DIR, render_dir, TEMPLATE_OUTPUT_ROOT_DIR)
                logging.info(f"Load 3D object mesh and rendering templates time: {time.time() - start_time} seconds")

                # 1st stage
                # Let's infer 2D object detections

                start_time = time.time()
                ISM.init_templates(
                    get_obj_template_dir(object_class_id, TEMPLATE_OUTPUT_ROOT_DIR), 
                    segmentator_model, device)
                logging.info(f"init_templates into segmentator_model time: {time.time() - start_time} seconds")


                obj_class_id_path = None
                if os.path.isdir(OUTPUT_DIR):
                    group_id_path = ALGORITHM_OUTPUT + "/" + f"{group_id:06d}"
                    scene_id_path = group_id_path + "/" + f"{scene_id:06d}"
                    camera_id_path = scene_id_path + "/" + str(camera_id) # no leading zeros for camera id numeric value
                    obj_class_id_path = camera_id_path + "/" + f"{object_class_id:06d}"
                    os.makedirs(obj_class_id_path, exist_ok=True)

                obj_class_detections = ISM.run_inference(
                    model=segmentator_model, 
                    device=device,
                    low_gpu_mem_mode=LOW_GPU_MEMORY_MODE,
                    output_dir=obj_class_id_path, 
                    cad_model=mesh,
                    rgb_image=color,
                    all_image_detections=all_image_detections,
                    query_decriptors=query_decriptors, 
                    query_appe_descriptors=query_appe_descriptors, 
                    depth_image=depth, 
                    cam_K=K, 
                    depth_scale=1.0,
                    min_detection_final_score=MIN_SEGMENTATION_FINAL_SCORE
                )
                        
                if True and obj_class_id_path is not None and len(obj_class_id_path) > 0:
                    ISM.save_output(obj_class_id_path, color, obj_class_detections, only_best_detection=False)

                if LOW_GPU_MEMORY_MODE:
                    ISM.unload_model_to_cpu(segmentator_model)
                    gc.collect()
                    torch.cuda.empty_cache()
                    pem.load_model_to_gpu()
                    gc.collect()
                    torch.cuda.empty_cache()

                
                # 2nd stage
                # Now that we have the 2D object detections, let's run pose inference on each detection.
                
                start_time = time.time()
                all_tem_pts, all_tem_feat = pem.get_templates(
                    os.path.join(get_obj_template_dir(object_class_id, TEMPLATE_OUTPUT_ROOT_DIR), "templates")
                )
                logging.info(f"get_templates time: {time.time() - start_time} seconds")

                start_time = time.time()
                model_points = pem.sample_points_from_mesh(mesh)
                input_data, detections = pem.prepare_test_data(obj_class_detections, color, depth, whole_pts, K, model_points)
                logging.info(f"sample_points_from_mesh and prepare_test_data time: {time.time() - start_time} seconds")

                start_time = time.time()
                pose_scores, pred_rot, pred_trans = pem.infer_pose(all_tem_pts, all_tem_feat, input_data)
                pred_trans *= 1000.0 # convert from meters back to millimeters
                logging.info(f"infer_pose time: {time.time() - start_time} seconds")

                pem.render_predictions(color, model_points, input_data, pose_scores, pred_rot, pred_trans, os.path.join(obj_class_id_path, "vis_pem.png"),
                                        only_highest_score = False, meters2mm=True)
                
                if LOW_GPU_MEMORY_MODE:
                    pem.unload_model_to_cpu()
                    gc.collect()
                    torch.cuda.empty_cache()
                    ISM.load_model_to_gpu(segmentator_model)
                    gc.collect()
                    torch.cuda.empty_cache()

                avg_total_inference_time.update(time.time() - start_whole_image_inference_time)
                logging.info(f"Avg. infer. time per image: {avg_total_inference_time.avg} seconds")
                logging.info(f"Total inferences made: {avg_total_inference_time.count}")
            
                # No need to iterate over object instances within each class.
                # SAM-6D detects all instances and infers their poses in a single pass, above.
                # object_instance_ids = object_instance_ids[0 : min(2, len(object_instance_ids)) ] # use only a couple of instances for each object class, just to test drive
                # for object_instance_id in object_instance_ids:
                #     #mask = reader.get_visible_object_mask(group_id, scene_id, camera_id, object_instance_id)
                #     pass

    total_time_overall = time.time() - start_time_overrall
    logging.info(f"Overall script time: {total_time_overall} seconds")
    logging.info(f"Avg inference image time: {(total_time_overall/total_inferences_made)} seconds")
