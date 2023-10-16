#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene
import os
import time
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state, pca_feature2image, get_pca_transform_matrix, get_similarity_labels, COLORS
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, instance_centers, pca_transform=None):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    feature_pca_path = os.path.join(model_path, name, "ours_{}".format(iteration), "feature_pca")
    seg_path = os.path.join(model_path, name, "ours_{}".format(iteration), "seg")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(feature_pca_path, exist_ok=True)
    makedirs(seg_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        render_pkg = render(view, gaussians, pipeline, background)
        rendering = render_pkg["render"]
        rendered_feature = render_pkg["instance_feature"]
        feature_pca = pca_feature2image(rendered_feature, v=pca_transform)
        seg_labels = get_similarity_labels(rendered_feature, instance_centers)
        seg_image = COLORS[seg_labels % COLORS.shape[0]] / 255.0
        seg_image = seg_image.permute(2, 0, 1)
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(feature_pca, os.path.join(feature_pca_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(seg_image, os.path.join(seg_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        print("Performing PCA on Gaussians' instance features...")
        feature_pca_transform = get_pca_transform_matrix(gaussians.get_instance_feature)

        # get instance feature centers
        t_start = time.time()
        instance_centers = gaussians.compute_instance_centers()
        t_end = time.time()
        print(f"Time elapsed for clustering: {t_end - t_start:.3f} s.")

        if not skip_train:
             render_set(dataset.model_path, "train", 
                        scene.loaded_iter, scene.getTrainCameras(), 
                        gaussians, pipeline, background,
                        instance_centers=instance_centers,
                        pca_transform=feature_pca_transform)

        if not skip_test:
             render_set(dataset.model_path, "test", 
                        scene.loaded_iter, scene.getTestCameras(), 
                        gaussians, pipeline, background,
                        instance_centers=instance_centers,
                        pca_transform=feature_pca_transform)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)