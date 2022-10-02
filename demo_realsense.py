"""Fuse 1000 RGB-D images from the 7-scenes dataset into a TSDF voxel volume with 2cm resolution.
"""
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np

import fusion


def main(voxel_size: float = 0.01, depth_trunc: float = 5.0):
    # ======================================================================================================== #
    # (Optional) This is an example of how to compute the 3D bounds
    # in world coordinates of the convex hull of all camera view
    # frustums in the dataset
    # ======================================================================================================== #
    print("Estimating voxel volume bounds...")
    imgs = ["d415/2022-09-25_16-52-05", "d435/2022-09-25_16-49-10"]
    n_imgs = len(imgs)
    vol_bnds = np.zeros((3, 2))

    for img in imgs:
        parent = Path(img).parent
        metadata = json.load(open(os.path.join(parent, "metadata.json")))

        cam_intr = np.array(metadata["intrinsics"]["intrinsic_matrix"])

        # Read depth image and camera pose
        depth_im = cv2.imread(f"{img}_depth.png", -1).astype(float)
        depth_im /= metadata[
            "depth_scale"
        ]  # depth is saved in 16-bit PNG in millimeters

        # Apply truncation to depth image, anything above 4m is bad
        depth_im[depth_im > depth_trunc] = 0

        extrinsincs = json.load(open(os.path.join(parent, "extrinsics.json")))
        cam_pose = np.array(extrinsincs["c2w"])  # 4x4 rigid transformation matrix

        # Compute camera view frustum and extend convex hull
        view_frust_pts = fusion.get_view_frustum(depth_im, cam_intr, cam_pose)
        vol_bnds[:, 0] = np.minimum(vol_bnds[:, 0], np.amin(view_frust_pts, axis=1))
        vol_bnds[:, 1] = np.maximum(vol_bnds[:, 1], np.amax(view_frust_pts, axis=1))

    print("vol_bnds", vol_bnds)

    # ======================================================================================================== #

    # ======================================================================================================== #
    # Integrate
    # ======================================================================================================== #
    # Initialize voxel volume
    print("Initializing voxel volume...")
    tsdf_vol = fusion.TSDFVolume(vol_bnds, voxel_size=voxel_size)

    # Loop through RGB-D images and fuse them together
    t0_elapse = time.time()
    for i, img in enumerate(imgs):
        print("Fusing frame %d/%d" % (i + 1, n_imgs))
        parent = Path(img).parent
        metadata = json.load(open(os.path.join(parent, "metadata.json")))

        cam_intr = np.array(metadata["intrinsics"]["intrinsic_matrix"])

        # Read RGB-D image and camera pose
        color_image = cv2.cvtColor(cv2.imread(f"{img}_color.png"), cv2.COLOR_BGR2RGB)
        depth_im = cv2.imread(f"{img}_depth.png", -1).astype(float)
        depth_im /= metadata[
            "depth_scale"
        ]  # depth is saved in 16-bit PNG in millimeters

        # Apply truncation to depth image, anything above 4m is bad
        depth_im[depth_im > depth_trunc] = 0

        extrinsincs = json.load(open(os.path.join(parent, "extrinsics.json")))
        cam_pose = np.array(extrinsincs["c2w"])  # 4x4 rigid transformation matrix

        # Integrate observation into voxel volume (assume color aligned with depth)
        tsdf_vol.integrate(color_image, depth_im, cam_intr, cam_pose, obs_weight=1.0)

    fps = n_imgs / (time.time() - t0_elapse)
    print("Average FPS: {:.2f}".format(fps))

    # Get mesh from voxel volume and save to disk (can be viewed with Meshlab)
    print("Saving mesh to mesh.ply...")
    verts, faces, norms, colors = tsdf_vol.get_mesh()
    fusion.meshwrite("mesh.ply", verts, faces, norms, colors)

    # Get point cloud from voxel volume and save to disk (can be viewed with Meshlab)
    print("Saving point cloud to pc.ply...")
    point_cloud = tsdf_vol.get_point_cloud()
    fusion.pcwrite("pc.ply", point_cloud)


if __name__ == "__main__":
    main()
