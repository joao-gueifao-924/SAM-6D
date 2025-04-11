
## SCRIPT ARGUMENTS ##

# set the paths
export ROOT_DIR=/home/joao/source/SAM-6D/SAM-6D
export CAD_PATH=$ROOT_DIR/Data/Example/obj_000005.ply    # path to a given cad model(mm)
export RGB_PATH=$ROOT_DIR/Data/Example/rgb.png           # path to a given RGB image
export DEPTH_PATH=$ROOT_DIR/Data/Example/depth.png       # path to a given depth map(mm)
export CAMERA_PATH=$ROOT_DIR/Data/Example/camera.json    # path to given camera intrinsics
export OUTPUT_DIR=$ROOT_DIR/Data/Example/outputs         # path to a pre-defined file for saving results

# Run instance segmentation model
export SEGMENTOR_MODEL=fastsam

# Allows inference on a GPU with at least 8 GB of RAM.
# Inference will be slower as some computations will be made on CPU instead of GPU and
# memory cleaning operations will be more frequent.
export LOW_GPU_MEM_MODE=1

# ===============================================================
# ===============================================================



echo "--- Stage 1: Render CAD templates ---"
cd Render
time blenderproc run render_custom_templates.py --custom-blender-path /home/joao/Downloads/blender-3.3.1-linux-x64 --output_dir $OUTPUT_DIR --cad_path $CAD_PATH #--colorize True 

echo "--- Finished Stage 1 ---"
echo ""

echo "--- Stage 2: Run instance segmentation model ---"

cd ../Instance_Segmentation_Model
time python run_inference_custom.py --segmentor_model $SEGMENTOR_MODEL --output_dir $OUTPUT_DIR --cad_path $CAD_PATH --rgb_path $RGB_PATH --depth_path $DEPTH_PATH --cam_path $CAMERA_PATH

echo "--- Finished Stage 2 ---"
echo ""


echo "--- Stage 3: Run pose estimation model ---"
export SEG_PATH=$OUTPUT_DIR/sam6d_results/detection_ism.json

cd ../Pose_Estimation_Model
time python run_inference_custom.py --output_dir $OUTPUT_DIR --cad_path $CAD_PATH --rgb_path $RGB_PATH --depth_path $DEPTH_PATH --cam_path $CAMERA_PATH --seg_path $SEG_PATH --low_gpu_memory_mode $LOW_GPU_MEM_MODE

echo "--- Finished Stage 3 ---"
echo "--- All stages complete ---"
