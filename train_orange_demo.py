#!/usr/bin/env python3
import sys
import os
import json
import random
import argparse
from concurrent.futures import ThreadPoolExecutor

import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from pytorch_msssim import ssim
from plyfile import PlyData
from scipy.ndimage import median_filter

from diffusers import StableDiffusionDepth2ImgPipeline

# Function for horizontal interpolation
from cross_section import interpolate_along_camera_direction

# Utils and Gaussian splatting
from utils.sh_utils import eval_sh
from scene.gaussian_model import GaussianModel
from utils.graphics_utils import BasicPointCloud
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.cameras import Camera as GSCamera
from gaussian_renderer import render
from utils.system_utils import searchForMaxIteration
from utils.decode_param import decode_param_json, load_params_from_gs
from utils.transformation_utils import (
    transform2origin, undotransform2origin,
    shift2center111, undoshift2center111,
    apply_cov_rotations, apply_inverse_cov_rotations,
    apply_inverse_rotations, generate_rotation_matrices
)
from utils.camera_view_utils import (
    get_center_view_worldspace_and_observant_coordinate,
    get_camera_view
)
from utils.render_utils import (
    initialize_rasterize, generate_plane, generate_plane_center, plane_filter
)
from utils.save_video import save_video
from utils.threestudio_utils import cleanup
from particle_filling.filling import *
from mpm_solver_warp.mpm_solver_warp import MPM_Simulator_WARP
import warp as wp
from sds_demo import one_step_sds_orange


def parse_args():
    parser = argparse.ArgumentParser(
        description="Gaussian Splatting Training Pipeline"
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Path to training JSON config file")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Directory containing gaussians checkpoints")
    parser.add_argument("--physics_config", type=str, required=True,
                        help="Scene physics JSON config path")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Directory for outputs (images, ply)")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug logging and PLY dumps")
    parser.add_argument("--gs_path", type=str, default=None,
                        help="Override training PLY path")
    parser.add_argument("--gs_ori_path", type=str, default=None,
                        help="Override original PLY path")
    return parser.parse_args()


def load_checkpoint(base_path, override_path=None):
    if override_path:
        ply_file = override_path
    else:
        cp_dir = os.path.join(base_path, "point_cloud")
        it = searchForMaxIteration(cp_dir)
        ply_file = os.path.join(cp_dir, f"iteration_{it}", "point_cloud.ply")
    gauss = GaussianModel(sh_degree=0)
    gauss.load_ply_zero_sh(ply_file)
    return gauss


def save_image(render, out_dir, frame, prefix="frame_"):
    arr = (render.permute(1,2,0).cpu().numpy() * 255).astype('uint8')
    img = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{prefix}{frame:04d}.png")
    cv2.imwrite(path, img)
    return path


def get_ssim_loss(pred, target):
    return 1 - ssim(pred.unsqueeze(0), target.unsqueeze(0), data_range=1, size_average=True)


def create_3d_grid(gauss, grid_size):
    xyz = gauss.get_xyz
    min_c = xyz.min(dim=0)[0]; max_c = xyz.max(dim=0)[0]
    dims = (max_c - min_c) / torch.tensor(grid_size, device=xyz.device)
    grid = {}
    for idx in range(xyz.size(0)):
        key = tuple(((xyz[idx]-min_c)/dims).floor().long().tolist())
        grid.setdefault(key, []).append(idx)
    return grid


def smooth_gaussians_in_grid(gauss, grid):
    feats = gauss.get_features
    out = torch.zeros_like(feats)
    for indices in grid.values():
        if not indices: continue
        out[indices] = feats[indices].mean(dim=0)
    gauss._features_dc.copy_(out)


def preprocess_particles(gauss, pipeline, prep_params, args):
    params = load_params_from_gs(gauss, pipeline)
    pos = params['pos']; cov = params['cov3D_precomp']; shs = params['shs']
    opacity = params['opacity']; screen_pts = params['screen_points']
    tpos, scale_o, mean_o = transform2origin(pos)
    tpos = shift2center111(tpos)
    cov = apply_cov_rotations(cov, rotation_matrices)
    cov = cov * (scale_o*scale_o)
    return shs, opacity, tpos.cuda(), cov, scale_o, mean_o, screen_pts


def training_step(gauss, loss, grad_mask=None, viewspace=None, vis_filter=None):
    for g in gauss.optimizer.param_groups:
        for p in g['params']:
            if p.grad is None: continue
            if grad_mask is not None:
                p.grad[~grad_mask] = 0
            if g.get('name','')=='opacity': p.grad.zero_()
    gauss.add_densification_stats(viewspace, vis_filter)
    gauss.optimizer.step()
    with torch.no_grad():
        gauss._opacity.copy_(init_opacity.clone().detach().fill_(max_opacity))
        gauss._scaling.clamp_(max=-16)
    gauss.optimizer.zero_grad()


def density_and_prune(gauss, epoch):
    if epoch>0 and epoch%cfg['density_every']==0:
        gauss.densify_and_prune(0.0002, min_opacity=1e-7, extent=4)
    if epoch>0 and epoch%cfg['voxel_smooth_every']==0:
        grid = create_3d_grid(gauss, cfg['voxel_grid_size'])
        smooth_gaussians_in_grid(gauss, grid)


def main():
    global cfg, init_opacity, max_opacity, rotation_matrices
    args = parse_args()
    cfg = json.load(open(args.config))
    os.makedirs(args.output_path, exist_ok=True)

    _, _, _, prep_params, camera_params = decode_param_json(args.physics_config)
    rotation_matrices = generate_rotation_matrices(
        torch.tensor(prep_params['rotation_degree']), prep_params['rotation_axis']
    )

    gauss = load_checkpoint(args.model_path, args.gs_path)
    gauss_ori = load_checkpoint(args.model_path, args.gs_ori_path)

    gauss.training_setup(argparse.Namespace(**cfg))
    init_opacity = load_params_from_gs(gauss, argparse.Namespace(**cfg))['opacity']
    max_opacity = init_opacity.max().item()
    gauss._opacity.copy_(init_opacity.clone().fill_(max_opacity))

    pipe_v = StableDiffusionDepth2ImgPipeline.from_pretrained(
        cfg['vertical_sd_model']).to('cuda:2')
    pipe_h = StableDiffusionDepth2ImgPipeline.from_pretrained(
        cfg['horizontal_sd_model']).to('cuda:1')

    transform = transforms.ToTensor()
    for epoch in range(cfg['total_epochs']):
        print(f"Epoch {epoch}/{cfg['total_epochs']}")
        density_and_prune(gauss, epoch)

        # VERTICAL
        for i in range(cfg['views_per_vertical_epoch']):
            ss, op, pos, cov, sc, mo, sp = preprocess_particles(
                gauss, argparse.Namespace(**cfg), prep_params, args
            )
            view_center, obs = get_center_view_worldspace_and_observant_coordinate(
                torch.tensor(camera_params['mpm_space_viewpoint_center']).cuda(),
                torch.tensor(camera_params['mpm_space_vertical_upward_axis']).cuda(),
                rotation_matrices, sc, mo
            )
            cur_cam, raw_cam = get_camera_view(
                args.model_path, default_camera_index=-1,
                center_view_world_space=view_center,
                observant_coordinates=obs,
                init_azimuthm=12*i, init_elevation=0,
                init_radius=camera_params['init_radius']
            )
            mask, _ = plane_filter(
                generate_plane(raw_cam, prep_params['particle_filling']['boundary']),
                pos, raw_cam, surf_dis=0.006
            )
            pos_cs, shs_cs, cov_cs, op_cs, sp_cs = pos[mask], ss[mask], cov[mask], op[mask], sp[mask]
            rasterize = initialize_rasterize(cur_cam, gauss, argparse.Namespace(**cfg), torch.tensor([1,1,1]).cuda(), 512,512)
            colors = eval_sh(shs_cs, cur_cam, gauss, pos_cs, None)
            rend, radii, _, _ = rasterize(
                means3D=pos_cs, means2D=sp_cs,
                shs=None, colors_precomp=colors,
                opacities=op_cs, scales=None,
                rotations=None, cov3D_precomp=cov_cs
            )
            if epoch % 30 == 0:
                init_img = save_image(rend, os.path.join(args.output_path,'vert_init'), i, 'v')
                ref = one_step_sds_orange(Image.open(init_img), rend, cfg['views_per_vertical_epoch'] - i, pipe_v, 'vertical')
            else:
                ref = Image.open(os.path.join(args.output_path,'vert_init', f'v{i:04d}.png'))
            tgt = transform(ref).to(rend.device)
            loss = 0.7 * get_ssim_loss(rend, tgt) + 0.3 * F.mse_loss(rend, tgt)
            loss.backward()
            training_step(gauss, loss, mask, sp, radii>0)

        # HORIZONTAL
        centers, avg_dis = interpolate_along_camera_direction(raw_cam, pos, cfg['horizontal_steps'])
        avg_dis = avg_dis.item()
        for idx, cen in enumerate(centers[10:60]):
            ss, op, pos, cov, sc, mo, sp = preprocess_particles(
                gauss, argparse.Namespace(**cfg), prep_params, args
            )
            cur_cam, raw_cam = get_camera_view(
                args.model_path, default_camera_index=-1,
                center_view_world_space=None, observant_coordinates=None,
                init_azimuthm=0, init_elevation=90,
                init_radius=camera_params['init_radius']
            )
            mask, _ = plane_filter(
                generate_plane_center(raw_cam, cen),
                pos, raw_cam, surf_dis=avg_dis/2
            )
            pos_cs, shs_cs, cov_cs, op_cs, sp_cs = pos[mask], ss[mask], cov[mask], op[mask], sp[mask]
            rasterize = initialize_rasterize(cur_cam, gauss, argparse.Namespace(**cfg), torch.tensor([1,1,1]).cuda(), 512,512)
            colors = eval_sh(shs_cs, cur_cam, gauss, pos_cs, None)
            rend, radii, _, _ = rasterize(
                means3D=pos_cs, means2D=sp_cs,
                shs=None, colors_precomp=colors,
                opacities=op_cs, scales=None,
                rotations=None, cov3D_precomp=cov_cs
            )
            if epoch % 30 == 0:
                init_img = save_image(rend, os.path.join(args.output_path,'horiz_init'), idx, 'h')
                ref = one_step_sds_orange(Image.open(init_img), rend, cfg['horizontal_steps'] - idx, pipe_h, 'horizontal')
            else:
                ref = Image.open(os.path.join(args.output_path,'horiz_init', f'h{idx:04d}.png'))
            tgt = transform(ref).to(rend.device)
            loss = 0.7 * get_ssim_loss(rend, tgt) + 0.3 * F.mse_loss(rend, tgt)
            loss.backward()
            training_step(gauss, loss, mask, sp, radii>0)

        # ORIGINAL RECON
        for t in range(30):
            # render original target
            ss_o, op_o, pos_o, cov_o, sc_o, mo_o, sp_o = preprocess_particles(
                gauss_ori, argparse.Namespace(**cfg), prep_params, args
            )
            az = random.randint(0, 360)
            el = random.randint(-90, 90)
            cur_cam, raw_cam = get_camera_view(
                args.model_path, default_camera_index=-1,
                center_view_world_space=None, observant_coordinates=None,
                init_azimuthm=az, init_elevation=el,
                init_radius=camera_params['init_radius']
            )
            cov_o = apply_inverse_cov_rotations(cov_o/(sc_o*sc_o), rotation_matrices)
            mask_o, _ = plane_filter(
                generate_plane(raw_cam, prep_params['particle_filling']['boundary']),
                pos_o, raw_cam, surf_dis=0.006
            )
            pos_cs_o = pos_o[mask_o]; shs_cs_o = ss_o[mask_o]; cov_cs_o = cov_o[mask_o]; op_cs_o = op_o[mask_o]; sp_cs_o = sp_o[mask_o]
            rasterize_o = initialize_rasterize(cur_cam, gauss_ori, argparse.Namespace(**cfg), torch.tensor([1,1,1]).cuda(), 512,512)
            rend_o, _, _, _ = rasterize_o(
                means3D=pos_cs_o, means2D=sp_cs_o,
                shs=None, colors_precomp=eval_sh(shs_cs_o, cur_cam, gauss_ori, pos_cs_o, None),
                opacities=op_cs_o, scales=None,
                rotations=None, cov3D_precomp=cov_cs_o
            )
            # student pass
            ss_s, op_s, pos_s, cov_s, sc_s, mo_s, sp_s = preprocess_particles(
                gauss, argparse.Namespace(**cfg), prep_params, args
            )
            mask_s, _ = plane_filter(
                generate_plane(raw_cam, prep_params['particle_filling']['boundary']),
                pos_s, raw_cam, surf_dis=0.006
            )
            pos_cs_s = pos_s[mask_s]; shs_cs_s = ss_s[mask_s]; cov_cs_s = cov_s[mask_s]; op_cs_s = op_s[mask_s]; sp_cs_s = sp_s[mask_s]
            rasterize_s = initialize_rasterize(cur_cam, gauss, argparse.Namespace(**cfg), torch.tensor([1,1,1]).cuda(), 512,512)
            rend_s, radii_s, _, _ = rasterize_s(
                means3D=pos_cs_s, means2D=sp_cs_s,
                shs=None, colors_precomp=eval_sh(shs_cs_s, cur_cam, gauss, pos_cs_s, None),
                opacities=op_cs_s, scales=None,
                rotations=None, cov3D_precomp=cov_cs_s
            )
            tgt = rend_o.detach()
            loss = 0.6 * get_ssim_loss(rend_s, tgt) + 0.4 * F.mse_loss(rend_s, tgt)
            loss.backward()
            training_step(gauss, loss, mask_s, sp_s, radii_s>0)

        # save intermediate model
        if epoch % 20 == 0:
            gauss.save_ply(os.path.join(args.output_path, f'model_ep{epoch}.ply'))

    print("Training complete.")

if __name__ == '__main__':
    main()
