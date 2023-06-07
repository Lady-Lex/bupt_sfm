import os
import pyvista
import gradio as gr
import plotly.graph_objs as go
from multiprocessing import Process, Queue

from .sfm_main import *
from .stream import *
from .config import *
from .api import api


def plot_ply_file(ply_save_dir: str):
    point_cloud = pyvista.read(os.path.join(ply_save_dir, "sfm_output.ply"))
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=point_cloud.points[:,0], y=point_cloud.points[:,1], z=point_cloud.points[:,2],
                mode='markers',
                marker=dict(
                    size=2,
                    color=['rgb({},{},{})'.format(r,g,b) for r,g,b in zip(point_cloud.active_scalars[:,0], point_cloud.active_scalars[:,1], point_cloud.active_scalars[:,2])],
                )
            )
        ],
        layout=dict(
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False)
            )
        ),
    )
    return fig


def run(ply_save_dir: str) -> None:
    cfg = load_config_from_api()

    queue = Queue(maxsize=10)

    if cfg["running_ros"]:
        reader = Process(target=run_ros_topic_stream, args=(cfg, queue))
    else:
        if os.path.isdir(cfg["running_image_dir"]):
            reader = Process(target=image_stream, args=(cfg, queue))
        else:
            raise Exception("Wrong image directory path!")

    print("SFM Start!")
    reader.start()

    sfm = sfm_runner(queue, cfg)
    total_cloud, total_color = sfm()
    fig = plot_ply_file(ply_save_dir)

    reader.terminate()

    return fig


def set_config_and_run(File_Path: str,
                  calib: str,
                  ply_save_dir: str,
                  viz: bool,
                  ros: bool,
                  robust: bool,
                  save_reconstruction: bool,
                  feature_root: bool,
                  use_adaptive_suppression: bool,
                  dpviewer_activated: bool,
                  BA_activated: bool,
                  stride: int,
                  skip: int,
                  img_downscale: int,
                  feature_distance_ratio: float,
                  sift_peak_threshold: float,
                  sift_edge_threshold: float,
                  img_height: int,
                  img_width: int,
                  feature_matched_min: int,
                  feature_min_frames: int,
                  feature_min_frames_panorama: int,
                  feature_process_size: int,
                  feature_process_size_panorama: int,
                  save_distance_thresh: float):

    api.config_dict = {"running_image_dir": File_Path,
                       "running_calib": calib,
                       "running_stride": int(stride),
                       "running_skip": int(skip),
                       "running_viz": viz,
                       "running_ros": ros,
                       "running_robust": robust,
                       "running_save_reconstruction": save_reconstruction,
                       "image_height": int(img_height),
                       "image_width": int(img_width),
                       "image_downscale": int(img_downscale),
                       "feature_matched_min": int(feature_matched_min),
                       "feature_distance_ratio": feature_distance_ratio,
                       "feature_root": feature_root,
                       "feature_min_frames": int(feature_min_frames),
                       "feature_min_frames_panorama": int(feature_min_frames_panorama),
                       "feature_process_size": int(feature_process_size),
                       "feature_process_size_panorama": int(feature_process_size_panorama),
                       "feature_use_adaptive_suppression": use_adaptive_suppression,
                       "sift_peak_threshold": sift_peak_threshold,
                       "sift_edge_threshold": sift_edge_threshold,
                       "dpvierwer_activated": dpviewer_activated,
                       "BA_activated": BA_activated,
                       "save_distance_thresh": save_distance_thresh}
    
    fig = run(ply_save_dir)

    return fig


def shoot_once():
    api.shoot_once.value = True

def stop_shoot():
    api.stop_shoot.value = True
    

with gr.Blocks() as sfm_webui:
    gr.Markdown("## Light Structure from Motion System")

    gr_inputs = [
                #  gr.components.Textbox(label="file path or ros image topic", value="/webcam/image_raw"),
                 gr.components.Textbox(label="file path or ros image topic", value="/home/lexington2002/Studyspace/机器视觉技术课程设计/作业/bupt_sfm/data/Herz-Jesus-P8/images"),
                 gr.components.Textbox(label="calibration file path or ros camera_info topic", value="/home/lexington2002/Studyspace/机器视觉技术课程设计/作业/bupt_sfm/data/Herz-Jesus-P8/images/intrinsics.txt"),
                #  gr.components.Textbox(label="calibration file path or ros camera_info topic", value="/webcam/camera_info"),
                 gr.components.Textbox(label="ply file save directory", value="/home/lexington2002/Studyspace/机器视觉技术课程设计/作业/bupt_sfm/pointcloud")
                 ]

    with gr.Row():
        gr_inputs.append(gr.Checkbox(label="viz", value=False))
        gr_inputs.append(gr.Checkbox(label="ros", value=False))
        gr_inputs.append(gr.Checkbox(label="robust", value=True))
        gr_inputs.append(gr.Checkbox(label="save_reconstruction", value=True))

    with gr.Row():
        gr_inputs.append(gr.Checkbox(label="feature_root", value=True))
        gr_inputs.append(gr.Checkbox(label="use_adaptive_suppression(deprecated)", value=False))
        gr_inputs.append(gr.Checkbox(label="dpvierwer_activated(deprecated)", value=False))
        gr_inputs.append(gr.Checkbox(label="BA_activated", value=False))

    gr_inputs.append(gr.Slider(1, 10, label="stride", value=1))
    gr_inputs.append(gr.Slider(0, 10, label="skip", value=0))
    gr_inputs.append(gr.Slider(1, 10, label="img_downscale", value=2))
    gr_inputs.append(gr.Slider(0, 1, label="feature_distance_ratio", value=0.7))
    gr_inputs.append(gr.Slider(0, 1, label="sift_peak_threshold", value=0.1))
    gr_inputs.append(gr.Slider(0, 20, label="sift_edge_threshold", value=10))

    with gr.Row():
        gr_inputs.append(gr.Number(label="img_height", value=2048))
        gr_inputs.append(gr.Number(label="img_width", value=3072))
        gr_inputs.append(gr.Number(label="feature_matched_min", value=100))
        gr_inputs.append(gr.Number(label="feature_min_frames", value=4000))

    with gr.Row():
        gr_inputs.append(gr.Number(label="feature_min_frames_panorama", value=1000))
        gr_inputs.append(gr.Number(label="feature_process_size", value=2048))
        gr_inputs.append(gr.Number(label="feature_process_size_panorama", value=4096))
        gr_inputs.append(gr.Number(label="save_distance_thresh", value=600.0))

    gr_output = gr.Plot()
    submit_button = gr.Button("submit")

    submit_button.click(set_config_and_run, inputs=gr_inputs, outputs=gr_output)

    gr_inputs2 = []
    gr_inputs3 = []
    with gr.Row():
        ros_shoot_button = gr.Button("ros shoot")
        shoot_finished_button = gr.Button("shoot finished")
    ros_shoot_button.click(shoot_once, inputs=gr_inputs2, outputs=gr_output)
    shoot_finished_button.click(stop_shoot, inputs=gr_inputs3, outputs=gr_output)
