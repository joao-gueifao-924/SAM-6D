#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Get the directory where this script file itself is located
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Define the Dockerfile path relative to the script directory
DOCKERFILE_PATH="$SCRIPT_DIR/dockerfile"

# Define the context path (the parent directory of the script directory)
CONTEXT_PATH=$(realpath "$SCRIPT_DIR/..")

# Change these variables to your needs

# Demo example provided with SAM-6D code project:
# CAD_FILE=obj_000005.ply         # file with a given cad model(mm)
# RGB_FILE=rgb.png                # file with a given RGB image
# DEPTH_FILE=depth.png            # file with a given depth map(mm)
# CAMERA_FILE=camera.json         # file to given camera intrinsics
# SEGMENTOR_MODEL=fastsam         # either 'sam' or 'fastsam'

OUTPUT_DIR=/algorithm_output      # path where to save results
ALGORITHM_OUTPUT_OUTSIDE_CONTAINER_DIR=/home/joao/Downloads/algorithm_output

EXAMPLE_OUTSIDE_CONTAINER_DIR=$(realpath "$SCRIPT_DIR/../SAM-6D/Data/Example") # path outside Docker container that will be mounted
EXAMPLE_INSIDE_CONTAINER_DIR="/SAM-6D/Data/Example"

IPD_DATASET_ROOT_FOLDER="/mnt/061A31701A315E3D/ipd-dataset/bpc_baseline/datasets"
OBJECT_MESH_DIR="/ipd/models"

# The name of the container that will be generated.
IMAGE_NAME="sam-6d"


# Build the image specifying the Dockerfile location and the context directory
DOCKER_BUILDKIT=1 docker buildx build --load -t $IMAGE_NAME -f "$DOCKERFILE_PATH" "$CONTEXT_PATH" --build-arg EXAMPLE_INSIDE_CONTAINER_DIR

echo EXAMPLE_OUTSIDE_CONTAINER_DIR: $EXAMPLE_OUTSIDE_CONTAINER_DIR
echo EXAMPLE_INSIDE_CONTAINER_DIR: $EXAMPLE_INSIDE_CONTAINER_DIR
echo IPD_DATASET_ROOT_FOLDER: $IPD_DATASET_ROOT_FOLDER
echo ALGORITHM_OUTPUT_OUTSIDE_CONTAINER_DIR: $ALGORITHM_OUTPUT_OUTSIDE_CONTAINER_DIR
echo OUTPUT_DIR: $OUTPUT_DIR


docker run \
  -it --rm --gpus all \
  -p 5678:5678 \
  -e EXAMPLE_INSIDE_CONTAINER_DIR=$EXAMPLE_INSIDE_CONTAINER_DIR \
  -e OUTPUT_DIR=$OUTPUT_DIR \
  -e OBJECT_MESH_DIR=$OBJECT_MESH_DIR \
  --name $IMAGE_NAME \
  -v $EXAMPLE_OUTSIDE_CONTAINER_DIR:$EXAMPLE_INSIDE_CONTAINER_DIR:ro \
  -v "$IPD_DATASET_ROOT_FOLDER:/ipd:ro" \
  -v "$ALGORITHM_OUTPUT_OUTSIDE_CONTAINER_DIR:$OUTPUT_DIR:rw" \
  $IMAGE_NAME
#   -e CAD_FILE=$CAD_FILE \
#   -e RGB_FILE=$RGB_FILE \
#   -e DEPTH_FILE=$DEPTH_FILE \
#   -e CAMERA_FILE=$CAMERA_FILE \
#   -e SEGMENTOR_MODEL=$SEGMENTOR_MODEL \
