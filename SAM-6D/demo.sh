###### START OF SCRIPT ARGUMENTS SECTION ######

EXAMPLE_DIR=$EXAMPLE_INSIDE_CONTAINER_DIR
# or set it to the following path if running this script directly 
# on your host, i.e, outside Docker:
#EXAMPLE_DIR="/home/joao/source/SAM-6D/SAM-6D/Data/Example"

CAD_PATH=$EXAMPLE_DIR/obj_000005.ply    # path to a given cad model(mm)
RGB_PATH=$EXAMPLE_DIR/rgb.png           # path to a given RGB image
DEPTH_PATH=$EXAMPLE_DIR/depth.png       # path to a given depth map(mm)
CAMERA_PATH=$EXAMPLE_DIR/camera.json    # path to a given camera intrinsics matrix
SEG_PATH=$OUTPUT_DIR/sam6d_results/detection_ism.json # path to the output of 2nd stage, used by 3rd stage

# path where to save results
if [ -z "$OUTPUT_DIR" ]; then # if empty or unset
  OUTPUT_DIR="$EXAMPLE_DIR/../output"
fi

if [ -z "$BLENDER_PATH" ]; then # if empty or unset
  BLENDER_PATH="/home/joao/Downloads/blender-4.2.1-linux-x64"
fi

# Run instance segmentation model
SEGMENTOR_MODEL=fastsam

# Allows inference on a GPU with at least 8 GB of RAM.
# Inference will be slower as some computations will be made on CPU instead of GPU and
# memory cleaning operations will be more frequent.
LOW_GPU_MEM_MODE=1

###### END OF SCRIPT ARGUMENTS SECTION ######

echo "Script arguments:"
echo OUTPUT_DIR: $OUTPUT_DIR
echo CAD_PATH: $CAD_PATH
echo RGB_PATH: $RGB_PATH
echo DEPTH_PATH: $DEPTH_PATH
echo CAMERA_PATH: $CAMERA_PATH
echo SEG_PATH: $SEG_PATH
echo LOW_GPU_MEM_MODE: $LOW_GPU_MEM_MODE


echo "--- Stage 1: Render CAD templates ---"
cd Render
time blenderproc run render_custom_templates.py \
                    --custom-blender-path $BLENDER_PATH \
                    --output_dir $OUTPUT_DIR \
                    --cad_path $CAD_PATH 
                    #--colorize True 

echo "--- Finished Stage 1 ---"
echo ""

echo "--- Stage 2: Run instance segmentation model ---"

cd ../Instance_Segmentation_Model
time python run_inference_custom.py \
                  --segmentor_model $SEGMENTOR_MODEL \
                  --output_dir $OUTPUT_DIR \
                  --cad_path $CAD_PATH \
                  --rgb_path $RGB_PATH \
                  --depth_path $DEPTH_PATH \
                  --cam_path $CAMERA_PATH \
                  --low_gpu_memory_mode $LOW_GPU_MEM_MODE

echo "--- Finished Stage 2 ---"
echo ""


echo "--- Stage 3: Run pose estimation model ---"

cd ../Pose_Estimation_Model
time python run_inference_custom.py \
      --output_dir $OUTPUT_DIR \
      --cad_path $CAD_PATH \
      --rgb_path $RGB_PATH \
      --depth_path $DEPTH_PATH \
      --cam_path $CAMERA_PATH \
      --seg_path $SEG_PATH \
      --low_gpu_memory_mode $LOW_GPU_MEM_MODE

echo "--- Finished Stage 3 ---"
echo "--- All stages complete ---"
