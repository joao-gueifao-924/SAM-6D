import subprocess, logging, os, glob

def render_object_templates(object_class_id, object_meshes_dir, render_dir, template_output_root_dir, blender_path):
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
        logging.error("Captured STDOUT:")
        logging.error(e.stdout) # stdout is captured if capture_output=True
        logging.error("Captured STDERR:") # <--- THIS IS OFTEN MOST IMPORTANT
        logging.error(e.stderr) # stderr is captured if capture_output=True
        raise RuntimeError(f"Failed to generate templates for object {object_class_id}.")

def get_obj_template_dir(object_class_id, template_output_root_dir):
    this_object_template_output_dir = template_output_root_dir + f"/obj_{object_class_id:06d}"
    return this_object_template_output_dir
