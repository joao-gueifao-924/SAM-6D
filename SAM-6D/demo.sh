
## SCRIPT ARGUMENTS ##


EXAMPLE_DIR=$EXAMPLE_INSIDE_CONTAINER_DIR
# or set it to the following path if running this script directly on your host, i.e, outside Docker
# EXAMPLE_DIR="SAM-6D/Data/Example"

CAD_PATH=$EXAMPLE_DIR/obj_000005.ply    # path to a given cad model(mm)
RGB_PATH=$EXAMPLE_DIR/rgb.png           # path to a given RGB image
DEPTH_PATH=$EXAMPLE_DIR/depth.png       # path to a given depth map(mm)
CAMERA_PATH=$EXAMPLE_DIR/camera.json    # path to a given camera intrinsics matrix

# path where to save results
if [ -z "$OUTPUT_DIR" ]; then # if emtpy or unset
  OUTPUT_DIR="$EXAMPLE_DIR/../output"
fi

# Run instance segmentation model
SEGMENTOR_MODEL=fastsam

# Allows inference on a GPU with at least 8 GB of RAM.
# Inference will be slower as some computations will be made on CPU instead of GPU and
# memory cleaning operations will be more frequent.
LOW_GPU_MEM_MODE=1

# ===============================================================
# ===============================================================



echo "--- Stage 1: Render CAD templates ---"
cd Render
time blenderproc run render_custom_templates.py --custom-blender-path $BLENDER_PATH --output_dir $OUTPUT_DIR --cad_path $CAD_PATH #--colorize True 

echo "--- Finished Stage 1 ---"
echo ""

echo "--- Stage 2: Run instance segmentation model ---"

cd ../Instance_Segmentation_Model
time python run_inference_custom.py --segmentor_model $SEGMENTOR_MODEL --output_dir $OUTPUT_DIR --cad_path $CAD_PATH --rgb_path $RGB_PATH --depth_path $DEPTH_PATH --cam_path $CAMERA_PATH

echo "--- Finished Stage 2 ---"
echo ""


echo "--- Stage 3: Run pose estimation model ---"
SEG_PATH=$OUTPUT_DIR/sam6d_results/detection_ism.json

cd ../Pose_Estimation_Model
time python run_inference_custom.py --output_dir $OUTPUT_DIR --cad_path $CAD_PATH --rgb_path $RGB_PATH --depth_path $DEPTH_PATH --cam_path $CAMERA_PATH --seg_path $SEG_PATH --low_gpu_memory_mode $LOW_GPU_MEM_MODE

echo "--- Finished Stage 3 ---"
echo "--- All stages complete ---"
