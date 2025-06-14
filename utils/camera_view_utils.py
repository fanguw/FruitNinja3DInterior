import os
import json
import numpy as np
import torch
from scene.cameras import Camera as GSCamera
from utils.graphics_utils import focal2fov


def generate_camera_rotation_matrix(camera_to_object, object_vertical_downward):
    camera_to_object = camera_to_object / np.linalg.norm(
        camera_to_object
    )  # last column
    # the second column of rotation matrix is pointing toward the downward vertical direction
    camera_y = (
        object_vertical_downward
        - np.dot(object_vertical_downward, camera_to_object) * camera_to_object
    )
    camera_y = camera_y / np.linalg.norm(camera_y)  # second column
    first_column = np.cross(camera_y, camera_to_object)
    R = np.column_stack((first_column, camera_y, camera_to_object))
    return R


# supply vertical vector in world space
def generate_local_coord(vertical_vector):
    vertical_vector = vertical_vector / np.linalg.norm(vertical_vector)
    horizontal_1 = np.array([1, 1, 1])
    if np.abs(np.dot(horizontal_1, vertical_vector)) < 0.01:
        horizontal_1 = np.array([0.72, 0.37, -0.67])
    # gram schimit
    horizontal_1 = (
        horizontal_1 - np.dot(horizontal_1, vertical_vector) * vertical_vector
    )
    horizontal_1 = horizontal_1 / np.linalg.norm(horizontal_1)
    horizontal_2 = np.cross(horizontal_1, vertical_vector)

    return vertical_vector, horizontal_1, horizontal_2


# scalar (in degrees), scalar (in degrees), scalar, vec3, mat33 = [horizontal_1; horizontal_2; vertical];  -> vec3
def get_point_on_sphere(azimuth, elevation, radius, center, observant_coordinates):
    canonical_coordinates = (
        np.array(
            [
                np.cos(azimuth / 180.0 * np.pi) * np.cos(elevation / 180.0 * np.pi),
                np.sin(azimuth / 180.0 * np.pi) * np.cos(elevation / 180.0 * np.pi),
                np.sin(elevation / 180.0 * np.pi),
            ]
        )
        * radius
    )

    return center + observant_coordinates @ canonical_coordinates


def get_camera_position_and_rotation(
    azimuth, elevation, radius, view_center, observant_coordinates
):
    # get camera position
    position = get_point_on_sphere(
        azimuth, elevation, radius, view_center, observant_coordinates
    )
    # get rotation matrix
    R = generate_camera_rotation_matrix(
        view_center - position, -observant_coordinates[:, 2]
    )
    return position, R


def get_current_radius_azimuth_and_elevation(
    camera_position, view_center, observesant_coordinates
):
    center2camera = -view_center + camera_position
    radius = np.linalg.norm(center2camera)
    dot_product = np.dot(center2camera, observesant_coordinates[:, 2])
    cosine = dot_product / (
        np.linalg.norm(center2camera) * np.linalg.norm(observesant_coordinates[:, 2])
    )
    elevation = np.rad2deg(np.pi / 2.0 - np.arccos(cosine))
    proj_onto_hori = center2camera - dot_product * observesant_coordinates[:, 2]
    dot_product2 = np.dot(proj_onto_hori, observesant_coordinates[:, 0])
    cosine2 = dot_product2 / (
        np.linalg.norm(proj_onto_hori) * np.linalg.norm(observesant_coordinates[:, 0])
    )

    if np.dot(proj_onto_hori, observesant_coordinates[:, 1]) > 0:
        azimuth = np.rad2deg(np.arccos(cosine2))
    else:
        azimuth = -np.rad2deg(np.arccos(cosine2))
    return radius, azimuth, elevation



def get_camera_view(
    model_path,
    default_camera_index=0,
    center_view_world_space=None,
    observant_coordinates=None,
    show_hint=False,
    init_azimuthm=None,
    init_elevation=None,
    init_radius=None,
    move_camera=False,
    current_frame=0,
    delta_a=0,
    delta_e=0,
    delta_r=0,
):
    """Load one of the default cameras for the scene."""
    cam_path = os.path.join(model_path, "cameras.json")
    with open(cam_path) as f:
        data = json.load(f)

        if show_hint:
            if default_camera_index < 0:
                default_camera_index = 0
            r, a, e = get_current_radius_azimuth_and_elevation(
                data[default_camera_index]["position"],
                center_view_world_space,
                observant_coordinates,
            )
            print("Default camera ", default_camera_index, " has")
            print("azimuth:    ", a)
            print("elevation:  ", e)
            print("radius:     ", r)
            print("Now exit program and set your own input!")
            exit()

        if default_camera_index > -1:
            raw_camera = data[default_camera_index]

        else:
            raw_camera = data[0]  # get data to be modified

            assert init_azimuthm is not None
            assert init_elevation is not None
            assert init_radius is not None

            if move_camera:
                assert delta_a is not None
                assert delta_e is not None
                assert delta_r is not None
                position, R = get_camera_position_and_rotation(
                    init_azimuthm + current_frame * delta_a,
                    init_elevation + current_frame * delta_e,
                    init_radius + current_frame * delta_r,
                    center_view_world_space,
                    observant_coordinates,
                )
            else:
                position, R = get_camera_position_and_rotation(
                    init_azimuthm,
                    init_elevation,
                    init_radius,
                    center_view_world_space,
                    observant_coordinates,
                )
            raw_camera["rotation"] = R.tolist()
            raw_camera["position"] = position.tolist()

        tmp = np.zeros((4, 4))
        tmp[:3, :3] = raw_camera["rotation"]
        tmp[:3, 3] = raw_camera["position"]
        tmp[3, 3] = 1
        C2W = np.linalg.inv(tmp)
        R = C2W[:3, :3].transpose()
        T = C2W[:3, 3]
        
        width = raw_camera["width"]
        height = raw_camera["height"]
        fovx = focal2fov(raw_camera["fx"], width)
        fovy = focal2fov(raw_camera["fy"], height)

        print("--------camera---------")
        print(raw_camera)

        return GSCamera(
            colmap_id=0,
            R=R,
            T=T,
            FoVx=fovx,
            FoVy=fovy,
            image=torch.zeros((3, height, width)),  # fake
            gt_alpha_mask=None,
            image_name="fake",
            uid=0,
        ), {"R": R, "T": T, "width": width, "height": height, "fx": raw_camera["fx"], "fy": raw_camera["fy"], "fovx": fovx, "fovy": fovy}


import torch
import numpy as np

def get_visible_gaussians(
    positions: torch.Tensor,
    camera_params: dict
) -> tuple:
    """
    Identifies visible Gaussians from the camera's perspective and returns a boolean mask
    along with a per-pixel mapping of the closest Gaussians.
    
    Parameters:
        positions (torch.Tensor or np.ndarray): 
            Array of shape [N, 3] representing the 3D coordinates of Gaussians.
        camera_params (dict): Dictionary containing camera parameters:
            - 'R' (torch.Tensor or np.ndarray): Rotation matrix [3, 3].
            - 'T' (torch.Tensor or np.ndarray): Translation vector [3].
            - 'width' (int): Image width in pixels.
            - 'height' (int): Image height in pixels.
            - 'fx' (float): Focal length in pixels along the x-axis.
            - 'fy' (float): Focal length in pixels along the y-axis.
            - 'fovx' (float): Field of view in the x-direction (degrees).
            - 'fovy' (float): Field of view in the y-direction (degrees).
            - 'near' (float, optional): Near clipping plane distance. Default is 0.1.
            - 'far' (float, optional): Far clipping plane distance. Default is 100.0.
    
    Returns:
        mask (torch.Tensor): Boolean tensor of shape [N], where True indicates a visible Gaussian.
        pixel_map (torch.Tensor): Long tensor of shape [height, width], where each element holds
                                   the index of the closest Gaussian. If a pixel has no corresponding
                                   Gaussian, it is set to -1.
    """
    # Determine the desired device (CUDA if available, else CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Ensure positions are PyTorch tensors and move to device
    if isinstance(positions, np.ndarray):
        positions = torch.from_numpy(positions)
    elif not isinstance(positions, torch.Tensor):
        raise TypeError("positions must be a torch.Tensor or np.ndarray")

    positions = positions.to(device)

    # Ensure positions are of type float32
    if positions.dtype != torch.float32:
        positions = positions.float()

    # Function to convert camera parameters to tensors and move to device
    def to_tensor(x):
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).float().to(device)
        elif isinstance(x, torch.Tensor):
            return x.float().to(device)
        else:
            raise TypeError("Camera parameters 'R' and 'T' must be torch.Tensor or np.ndarray")

    # Unpack and convert camera parameters
    R = to_tensor(camera_params['R'])                # Rotation matrix [3, 3]
    T = to_tensor(camera_params['T']).view(1, 3)     # Translation vector [1, 3]
    fx = camera_params['fx']                         # Focal length x
    fy = camera_params['fy']                         # Focal length y
    width = camera_params['width']                   # Image width
    height = camera_params['height']                 # Image height
    fovx = camera_params['fovx']                     # Field of view x in degrees
    fovy = camera_params['fovy']                     # Field of view y in degrees
    near = camera_params.get('near', 0.1)            # Near clipping plane
    far = camera_params.get('far', 100.0)            # Far clipping plane

    N = positions.shape[0]  # Number of Gaussians

    # Step 1: Transform positions to camera coordinates
    # P_camera = R * P_world + T
    # positions: [N, 3], R: [3,3], T: [1,3]
    positions_camera = (R @ positions.T).T + T  # [N, 3]

    # Step 2: Compute depth (Z-coordinate in camera space)
    depths = positions_camera[:, 2]  # [N]

    # Step 3: Project to image plane
    # Avoid division by zero by replacing zeros or negative depths with a small epsilon
    epsilon = 1e-6
    depths_safe = torch.where(depths > 0, depths, torch.full_like(depths, epsilon))

    x_proj = (positions_camera[:, 0] / depths_safe) * fx + (width / 2)
    y_proj = (positions_camera[:, 1] / depths_safe) * fy + (height / 2)

    # Round to nearest pixel indices
    x_pix = torch.round(x_proj).long()
    y_pix = torch.round(y_proj).long()

    # Step 4: Filter Gaussians within image bounds and depth range
    within_depth = (depths > near) & (depths < far)
    within_width = (x_pix >= 0) & (x_pix < width)
    within_height = (y_pix >= 0) & (y_pix < height)
    within_fov = within_depth & within_width & within_height

    # Apply mask
    valid_indices = torch.nonzero(within_fov, as_tuple=False).squeeze()  # [M]
    if valid_indices.numel() == 0:
        # No valid Gaussians
        mask = torch.zeros(N, dtype=torch.bool, device=device)
        pixel_map = -torch.ones((height, width), dtype=torch.long, device=device)
        return mask, pixel_map

    # Handle case when only one valid index
    if valid_indices.dim() == 0:
        valid_indices = valid_indices.unsqueeze(0)

    # Filtered data
    depths_valid = depths[within_fov]      # [M]
    x_pix_valid = x_pix[within_fov]        # [M]
    y_pix_valid = y_pix[within_fov]        # [M]
    indices_valid = valid_indices          # Original indices [M]

    # Step 5: Sort Gaussians by depth ascending (closest first)
    sorted_depths, sorted_order = torch.sort(depths_valid)
    x_sorted = x_pix_valid[sorted_order]
    y_sorted = y_pix_valid[sorted_order]
    indices_sorted = indices_valid[sorted_order]

    # Step 6: Convert 2D pixel coordinates to unique 1D indices
    pixel_indices = y_sorted * width + x_sorted  # [M]

    # Step 7: Find unique pixels, keeping the first occurrence (closest Gaussian)
    # Since torch.unique with 'return_indices' is not available, we emulate it:
    # Compare each pixel with the previous one to find where a new unique pixel starts
    # Assumes pixel_indices is sorted by depth ascending
    # So the first occurrence of each unique pixel is the closest Gaussian

    # Create a shifted version of pixel_indices
    # Initialize with a value that cannot be a valid pixel index
    shifted_value = -1
    shifted_pixel_indices = torch.cat([torch.full((1,), shifted_value, dtype=pixel_indices.dtype, device=device),
                                       pixel_indices[:-1]])

    # Identify where the current pixel is different from the previous one
    unique_mask = pixel_indices != shifted_pixel_indices

    # Get the indices where unique_mask is True
    unique_indices = torch.nonzero(unique_mask, as_tuple=False).squeeze()

    # Step 8: Get the indices of the closest Gaussians
    visible_sorted_indices = indices_sorted[unique_indices]  # [P]

    # Step 9: Create a boolean mask with True for visible Gaussians
    mask = torch.zeros(N, dtype=torch.bool, device=device)
    mask[visible_sorted_indices] = True

    # Step 10: Create a per-pixel mapping of closest Gaussians
    # Initialize pixel_map with -1 indicating no Gaussian assigned
    pixel_map = -torch.ones((height, width), dtype=torch.long, device=device)

    # Assign the closest Gaussian index to each pixel
    # Compute 2D indices
    y_unique = y_sorted[unique_indices]
    x_unique = x_sorted[unique_indices]
    pixel_map[y_unique, x_unique] = visible_sorted_indices

    return mask, pixel_map
