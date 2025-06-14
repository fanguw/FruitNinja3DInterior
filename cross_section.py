import torch
from scene.cameras import Camera as GSCamera
import numpy as np
from PIL import Image

import taichi as ti

#ti.init(arch=ti.cuda, device_memory_GB=8.0)

def generate_plane(camera, boundary):
    # Extract camera pose parameters
    cam_rot = camera["R"]

    # Compute camera direction vector (looking direction)
    cam_dir = cam_rot @ np.array([0, 0, 1])  # Assuming camera looks in +z direction
    cam_dir /= np.linalg.norm(cam_dir)  # Normalize camera direction vector

    # Unpack object boundaries
    xmin, xmax, ymin, ymax, zmin, zmax = boundary

    # Choose a point on the plane (within object's boundary)
    plane_point = np.array([
        1,
        1,
        1,
    ])

    # Plane normal is the negative camera direction (plane faces towards the camera)
    plane_normal = -cam_dir

    # Plane equation: ax + by + cz + d = 0
    a, b, c = plane_normal
    d = -np.dot(plane_normal, plane_point)

    #print("plane:")
    #print(a, b, c, d)
    #print("camera:")
    #print(cam_rot)

    return [a, b, c, d]

def generate_plane_with_center(camera, boundary, center):
    # Extract camera pose parameters
    cam_rot = camera["R"]

    # Compute camera direction vector (looking direction)
    cam_dir = cam_rot @ np.array([0, 0, 1])  # Assuming camera looks in +z direction
    cam_dir /= np.linalg.norm(cam_dir)  # Normalize camera direction vector

    # Unpack object boundaries
    xmin, xmax, ymin, ymax, zmin, zmax = boundary

    # Choose a point on the plane (within object's boundary)
    plane_point = np.array([
        center[0],
        center[1],
        center[2]
    ])

    # Plane normal is the negative camera direction (plane faces towards the camera)
    plane_normal = -cam_dir

    # Plane equation: ax + by + cz + d = 0
    a, b, c = plane_normal
    d = -np.dot(plane_normal, plane_point)

    #print("plane:")
    #print(a, b, c, d)
    #print("camera:")
    #print(cam_rot)

    return [a, b, c, d]

def interpolate_along_camera_direction(camera, object_positions, steps=10):
    # Extract the camera rotation matrix and compute the camera direction
    cam_rot = camera["R"]
    camera_direction = cam_rot @ np.array([0, 0, 1])  # Assuming camera looks in +z direction
    camera_direction = camera_direction / np.linalg.norm(camera_direction)  # Normalize

    # Project all points onto the camera direction
    projections = object_positions.cpu().detach().numpy() @ camera_direction

    # Find the minimum and maximum projection values
    min_projection = np.min(projections)
    max_projection = np.max(projections)

    # Find the closest and furthest points in terms of projections
    closest_point = object_positions[np.argmin(projections)]
    furthest_point = object_positions[np.argmax(projections)]

    # Generate interpolated points
    interpolated_projections = np.linspace(min_projection, max_projection, steps)
    interpolated_points = np.outer(interpolated_projections, camera_direction)

    diffs = np.diff(interpolated_points, axis=0)
    distances = np.linalg.norm(diffs, axis=1)
    average_distance = np.mean(distances)

    return closest_point, furthest_point, interpolated_points, average_distance

def interpolate_along_camera_direction_return_distances(camera, object_positions, steps=10):
    # Extract the camera rotation matrix and compute the camera direction
    cam_rot = camera["R"]
    camera_direction = cam_rot @ np.array([0, 0, 1])  # Assuming camera looks in +z direction
    camera_direction = camera_direction / np.linalg.norm(camera_direction)  # Normalize

    # Project all points onto the camera direction
    projections = object_positions.cpu().detach().numpy() @ camera_direction

    # Find the minimum and maximum projection values
    min_projection = np.min(projections)
    max_projection = np.max(projections)

    # Find the closest and furthest points in terms of projections
    closest_point = object_positions[np.argmin(projections)]
    furthest_point = object_positions[np.argmax(projections)]

    # Generate interpolated points
    interpolated_projections = np.linspace(min_projection, max_projection, steps)
    interpolated_points = np.outer(interpolated_projections, camera_direction)

    # Calculate the step distances between consecutive interpolated points
    step_distances = np.linalg.norm(np.diff(interpolated_points, axis=0), axis=1)

    return closest_point, furthest_point, interpolated_points, step_distances

def generate_plane_center(camera, center):
    # Extract camera pose parameters
    cam_rot = camera["R"]

    # Compute camera direction vector (looking direction)
    cam_dir = cam_rot @ np.array([0, 0, 1])  # Assuming camera looks in +z direction
    cam_dir /= np.linalg.norm(cam_dir)  # Normalize camera direction vector

    # Plane normal is the negative camera direction (plane faces towards the camera)
    plane_normal = -cam_dir

    # Plane equation: ax + by + cz + d = 0
    a, b, c = plane_normal
    d = -np.dot(plane_normal, center)

    return [a, b, c, d]

def plane_filter(plane, pos, camera, surf_dis=0.01, include_double=False):
    a, b, c, d = plane
    # Define the normal vector (a, b, c) of the plane
    normal = torch.tensor([a, b, c], dtype=torch.float32, device=pos.device)

    # Convert camera rotation matrix (numpy array) to PyTorch tensor
    cam_rot = torch.tensor(camera["R"], dtype=torch.float32, device=pos.device)
    cam_trans = torch.tensor(camera["T"], dtype=torch.float32, device=pos.device)

    normal_camera = torch.matmul(cam_rot, normal)
    # Compute the distance of each point from the plane
    distances = torch.matmul(pos, normal) + d
    # Calculate the signed distance from camera to plane
    camera_dis = torch.dot(cam_trans, normal) + d

    """
    print("camera dis to the plane:")
    print(camera_dis)

    print("cam trans")
    print(cam_trans)
    """

    threshold = surf_dis
    camera_facing = torch.sign(camera_dis)
    mask = distances < 0
    """
    if camera_facing > 0:
        # Camera is viewing the side of the plane where points should be above (positive side)
        mask = distances < 0
    elif camera_facing < 0:
        # Camera is viewing the side of the plane where points should be below (negative side)
        mask = distances > 0
    else:
        print("something is off!!!!")
    """
    if include_double:
        mask_suf = torch.abs(distances) < threshold
    else:
        mask_suf = torch.logical_and(torch.abs(distances) < threshold, distances < 0)
    return mask, mask_suf

def plane_dist(plane, pos):
    a, b, c, d = plane
    normal = torch.tensor([a, b, c], dtype=torch.float32, device=pos.device)
    distances = torch.matmul(pos, normal) + d
    return distances


def compute_camera_params(boundary, image_width, image_height, fx, fy):
    xmin, xmax, ymin, ymax, zmin, zmax = boundary
    
    # Calculate the center of the object
    center = np.array([(xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2])

    # Calculate the size of the object (bounding box dimensions)
    size = np.array([xmax - xmin, ymax - ymin, zmax - zmin])

    # Set a reasonable camera distance (twice the maximum dimension of the object)
    camera_distance = np.linalg.norm(size) * 2

    # Camera position (move the camera back along the z-axis)
    T = center + np.array([0, 0, camera_distance])

    # Camera rotation (identity matrix for no rotation)
    R = np.eye(3)

    # Field of view in degrees (adjust as needed)
    fovx = 60  # Horizontal FoV
    fovy = 45  # Vertical FoV

    # Compute the focal lengths
    fx = image_width / (2 * np.tan(np.deg2rad(fovx) / 2))
    fy = image_height / (2 * np.tan(np.deg2rad(fovy) / 2))

    return R, T, fx, fy


