
# THIS IS A CODE COPY FROM IPDReader class inside FoundationPose project.
# Do not change it!!
# SEE https://github.com/joao-gueifao-924/FoundationPose/issues/13

import numpy as np
import json, os, random, glob, trimesh, os
import cv2


class IpdReader:
    def __init__(self, root_folder="/ipd", shorter_side=None):
      self.root_folder = root_folder
      self.train_folder = f"{root_folder}/train_pbr"
      self.mesh_folder = f"{root_folder}/models"
      self.scene_camera = {}
      self.scene_gt = {}
      self.scene_gt_info = {}
      self.shorter_side = shorter_side
      self.object_meshes, self.models_info = IpdReader.load_object_meshes(self.mesh_folder)

      # Preload camera and ground truth info for each camera, for all groups and scenes
      for cam in [1, 2, 3]:
        self.scene_camera[cam] = self._load_json_data(f"scene_camera_cam{cam}.json")
        self.scene_gt[cam] = self._load_json_data(f"scene_gt_cam{cam}.json")
        self.scene_gt_info[cam] = self._load_json_data(f"scene_gt_info_cam{cam}.json")

      # Determine downscale factor if target image shorter_side is provided
      if shorter_side is not None:
        sample_img_path = self._get_filepath(*self.get_random_dataset_point(), "rgb")
        sample_img = cv2.imread(sample_img_path, cv2.IMREAD_UNCHANGED)
        self.H, self.W = sample_img.shape[:2]
        self.downscale = shorter_side / min(self.H, self.W)
        self.H = int(self.H * self.downscale)
        self.W = int(self.W * self.downscale)

        # Update intrinsic camera matrix K for each group, scene and camera:
        for cam in [1, 2, 3]:
          for group_id, scenes in self.scene_camera[cam].items():
            for scene_id, cam_data in scenes.items():
              cam_data["cam_K"][:2] *= self.downscale
      else:
        self.downscale = 1

    @staticmethod
    def load_object_meshes(mesh_folder, load_info=False):
      """Load all object meshes from the models directory."""
      object_meshes = {}
      models_info = {}
      try:
        # Load models_info.json first
        if load_info:
          models_info_path = os.path.join(mesh_folder, "models_info.json")
          with open(models_info_path, 'r') as f:
            raw_models_info = json.load(f)
            # Convert object class ID keys from strings to integers
            models_info = {int(k): v for k, v in raw_models_info.items()}
        
        # Load each object mesh
        mesh_files = glob.glob(os.path.join(mesh_folder, "obj_*.ply"))
        for mesh_file in mesh_files:
          # Extract object ID from filename (e.g., "obj_000001.ply" -> 1)
          obj_id = int(os.path.basename(mesh_file).split('_')[1].split('.')[0])
          try:
            mesh = trimesh.load(mesh_file)
            # Convert vertices from millimeters to meters
            mesh.vertices = mesh.vertices / 1000.0
            object_meshes[obj_id] = mesh
          except Exception as e:
            print(f"Failed to load mesh for object {obj_id}: {e}")
      except Exception as e:
        print(f"Error loading object meshes: {e}")
      return object_meshes, models_info

    def _load_json_data(self, filename: str) -> dict:
        """Helper to load JSON data into memory."""
        data = {}
        for group_dir in os.listdir(self.train_folder):
            group_path = os.path.join(self.train_folder, group_dir)
            file_path = os.path.join(group_path, filename)
            if os.path.isfile(file_path):
                with open(file_path, 'r') as f:
                    raw_data = json.load(f)
                    group_id = int(group_dir)
                    data[group_id] = {
                        int(scene_id): self._process_scene_data(scene_data)
                        for scene_id, scene_data in raw_data.items()
                    }
        return data

    def _process_scene_data(self, scene_data: dict) -> dict:
        """Helper to process individual scene data."""
        if "cam_K" in scene_data:
            scene_data["cam_K"] = np.array(scene_data["cam_K"]).reshape(3, 3)
        if "cam_R_w2c" in scene_data:
            scene_data["cam_R_w2c"] = np.array(scene_data["cam_R_w2c"]).reshape(3, 3)
        if "cam_t_w2c" in scene_data:
            scene_data["cam_t_w2c"] = np.array(scene_data["cam_t_w2c"]).reshape(3, 1)
        if isinstance(scene_data, list):  # Handle lists of objects
            for obj_data in scene_data:
                if "cam_R_m2c" in obj_data:
                    obj_data["cam_R_m2c"] = np.array(obj_data["cam_R_m2c"]).reshape(3, 3)
                if "cam_t_m2c" in obj_data:
                    obj_data["cam_t_m2c"] = np.array(obj_data["cam_t_m2c"]).reshape(3, 1)
        return scene_data

    def _get_filepath(self, group: int, scene: int, camera: int, img_type: str, object_instance_id: int = None) -> str:
      """Helper to construct image file paths following the dataset structure."""
      path = f"{self.train_folder}/{group:06d}/{img_type}_cam{camera}/{scene:06d}"

      if object_instance_id is not None:
        path += f"_{object_instance_id:06d}.png"
      else:
        path += ".jpg" if img_type == "rgb" else ".png"
      return path
    
    def get_rgb_image(self, group: int, scene: int, camera: int) -> np.array:
        """Returns RGB (grayscale) image as numpy array."""
        filepath = self._get_filepath(group, scene, camera, "rgb")
        img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
        if self.downscale != 1:
            img = cv2.resize(img, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
        return img

    def get_depth_image(self, group: int, scene: int, camera: int) -> np.array:
        """Returns depth image with values in meters."""
        filepath = self._get_filepath(group, scene, camera, "depth")
        depth = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
        if self.downscale != 1:
            depth = cv2.resize(depth, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
        depth_scale = self.scene_camera[camera][group][scene]["depth_scale"]
        depth = depth.astype(np.float32) * depth_scale / 1000.0  # Convert to meters
        return depth

    def get_aolp_image(self, group: int, scene: int, camera: int) -> np.array:
        """Returns Angle of Linear Polarization image."""
        filepath = self._get_filepath(group, scene, camera, "aolp")
        img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
        if self.downscale != 1:
            img = cv2.resize(img, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
        return img

    def get_dolp_image(self, group: int, scene: int, camera: int) -> np.array:
        """Returns Degree of Linear Polarization image."""
        filepath = self._get_filepath(group, scene, camera, "dolp")
        img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
        if self.downscale != 1:
            img = cv2.resize(img, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
        return img

    def get_whole_object_mask(self, group: int, scene: int, camera: int, object_instance_id: int) -> np.array:
      """Returns binary mask including occluded parts of a specific object instance."""
      filepath = self._get_filepath(group, scene, camera, "mask", object_instance_id)
      mask = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
      if self.downscale != 1:
        mask = cv2.resize(mask, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
      return mask > 0  # Convert to boolean array

    def get_visible_object_mask(self, group: int, scene: int, camera: int, object_instance_id: int) -> np.array:
      """Returns binary mask of only visible parts of a specific object instance."""
      filepath = self._get_filepath(group, scene, camera, "mask_visib", object_instance_id)
      mask = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
      if self.downscale != 1:
        mask = cv2.resize(mask, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
      return mask > 0  # Convert to boolean array

    def get_camera_intrinsics_K_matrix(self, group: int, scene: int, camera: int) -> np.array:
        """Returns the intrinsic camera matrix K for the specified camera."""
        K = self.scene_camera[camera][group][scene]["cam_K"]
        return K

    def get_camera_pose(self, group: int, scene: int, camera: int) -> tuple:
        """Returns the camera rotation matrix and translation vector in world-to-camera coordinates."""
        cam_data = self.scene_camera[camera][group][scene]
        R = np.array(cam_data["cam_R_w2c"])
        t = np.array(cam_data["cam_t_w2c"]) / 1000.0  # Convert to meters
        return R, t

    def get_random_dataset_point(self) -> tuple:
      """
      Returns a (group_id, scene_id, camera_id) existent in the dataset.
      Useful to get random images from the dataset.
      """
      group_id = random.choice(list(self.scene_camera[1].keys()))
      camera_id = random.choice([1, 2, 3])
      scene_id = random.choice(list(self.scene_camera[camera_id][group_id].keys()))
      return group_id, scene_id, camera_id
    
    def get_object_instances(self, object_class_id: int, group_id: int, scene_id: int, camera_id: int) -> list:
      """
      Returns a list of object instance IDs for a given object class ID in the specified group, scene, and camera.
      """
      instances = []
      if group_id in self.scene_gt[camera_id] and scene_id in self.scene_gt[camera_id][group_id]:
        for instance_id, obj_data in enumerate(self.scene_gt[camera_id][group_id][scene_id]):
          if obj_data["obj_id"] == object_class_id:
            instances.append(instance_id)
      return instances
    
    def get_object_pose(self, object_instance_id: int, group_id: int, scene_id: int, camera_id: int) -> tuple:
      """
      Returns the (R, t) pose of an object_instance_id present in (group_id, scene_id, camera_id).
      """
      if group_id in self.scene_gt[camera_id] and scene_id in self.scene_gt[camera_id][group_id]:
        obj_data = self.scene_gt[camera_id][group_id][scene_id][object_instance_id]
        R = obj_data["cam_R_m2c"]
        t = obj_data["cam_t_m2c"] / 1000.0  # Convert to meters
        return R, t
      else:
        raise ValueError(f"Object instance {object_instance_id} not found in group {group_id}, scene {scene_id}, camera {camera_id}.")

    def enumerate_groups(self) -> list:
      """
      Returns a list of all group IDs present in the dataset.
      """
      return list(self.scene_camera[1].keys())

    def enumerate_scenes(self, group_id: int) -> list:
      """
      Returns a list of all scene IDs present in a given group.
      """
      if group_id in self.scene_camera[1]:
        return list(self.scene_camera[1][group_id].keys())
      else:
        raise ValueError(f"Group ID {group_id} not found in the dataset.")

    def enumerate_objects(self, group_id: int, scene_id: int, camera_id: int) -> dict:
      """
      Given (group_id, scene_id, camera_id), enumerate object classes and respective instances present.
      Returns a dictionary where keys are object classes that are present, and values are lists of object_instance_ids.
      """
      if group_id in self.scene_gt[camera_id] and scene_id in self.scene_gt[camera_id][group_id]:
        object_instances = {}
        for instance_id, obj_data in enumerate(self.scene_gt[camera_id][group_id][scene_id]):
          obj_id = obj_data["obj_id"]
          if obj_id not in object_instances:
            object_instances[obj_id] = []
          object_instances[obj_id].append(instance_id)
        return object_instances
      else:
        raise ValueError(f"Scene ID {scene_id} in group {group_id} not found for camera {camera_id}.")
      

    def get_object_mesh(self, object_class_id: int) -> trimesh.Trimesh:
        """
        Returns the 3D mesh for the specified object class ID.
        """
        if object_class_id not in self.object_meshes:
            raise KeyError(f"No mesh found for object class ID {object_class_id}")
        return self.object_meshes[object_class_id]
