# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers
import numpy as np
import torch
from pathlib import Path
import ffmpeg
import cv2

import sys
sys.path.append('../')
from motionsynth_code.motion import bvh

def render_joint_cv2_forvistest(data, skeleton,joints, fps, output, audiopath=None,color={'0' : (0,256,0)},parts = 'full_body'):
    #anim = bvh.load(bvhfile)[0]
    skl = skeleton
    positions_3d = data
    radius = torch.max(skl.offsets()).item() * 5

    frame_total = len(positions_3d)
    
    fps = fps
    W = 400
    H = 400
    imgsize = (W, H)
    filepath = output
    codec = cv2.VideoWriter_fourcc('m','p','4','v')
    writer = cv2.VideoWriter(filepath, codec, fps, imgsize)
    positions_3d[:,:,2] = -1 * positions_3d[:,:,2] + radius
    positions_3d = positions_3d * 250/radius
    positions_3d=np.round(positions_3d).astype(int)
    root_offset = positions_3d[0,0]
    delta_y = 200 - root_offset[2]
    delta_x = 200 - root_offset[0]
    positions_3d[:,:,2] += delta_y
    positions_3d[:,:,0] += delta_x
    #cv2.namedWindow('window')
    for frame in range(frame_total):
        img = np.full((400, 400, 3), 1, dtype=np.uint8)
        for j in joints:
            if (skl.parents()[j] in joints)&(skl.parents()[j] >= 0):
                x1 = positions_3d[frame,j,[0,2]]
                x2 = positions_3d[frame,skl.parents()[j],[0,2]]
                cv2.line(img, pt1=tuple(x1), pt2=tuple(x2), color=color,
                        thickness=1, lineType=cv2.LINE_AA, shift=0)
        writer.write(img)
    writer.release()

    if audiopath:
        outputpath = str(Path(output).parent) + '/' + str(Path(output).stem) + f'_audio_{parts}.mp4'
        instream_v = ffmpeg.input(output)
        instream_a = ffmpeg.input(audiopath)
        stream = ffmpeg.output(instream_v, instream_a, outputpath, vcodec='copy', acodec='aac')
        ffmpeg.run(stream)



def render_joint_cv2(data, skeleton,joints, fps, output, audiopath=None,color={'0' : (0,256,0)},parts = 'full_body'):
    #anim = bvh.load(bvhfile)[0]
    skl = skeleton
    positions_3d = data
    radius = torch.max(skl.offsets()).item() * 1

    #positions_3d[:,:,0] *= -1

    #normalize

    frame_total = len(positions_3d['0'])#viz_test以外はこれ使う
    
    fps = fps
    W = 400
    H = 400
    imgsize = (W, H)
    filepath = output
    codec = cv2.VideoWriter_fourcc('m','p','4','v')
    writer = cv2.VideoWriter(filepath, codec, fps, imgsize)
    for i in range(len(positions_3d)):
        positions_3d[f'{i}'][:,:,2] = -1 * positions_3d[f'{i}'][:,:,2] + radius
        positions_3d[f'{i}'] = positions_3d[f'{i}'] * 250/radius
        positions_3d[f'{i}']=np.round(positions_3d[f'{i}']).astype(int)
        root_offset = positions_3d[f'{i}'][0,0]
        delta_y = 200 - root_offset[2]
        delta_x = 200 - root_offset[0]
        positions_3d[f'{i}'][:,:,2] += delta_y
        positions_3d[f'{i}'][:,:,0] += delta_x
    #cv2.namedWindow('window')
    for frame in range(frame_total):
        img = np.full((400, 400, 3), 1, dtype=np.uint8)
        for i in range(len(positions_3d)):
            # print(i)
            for j in joints:
                if (skl.parents()[j] in joints)&(skl.parents()[j] >= 0):
                    # if i != len(positions_3d)-1:
                    x1 = positions_3d[f'{i}'][frame,joints.index(j),[0,2]]
                    x2 = positions_3d[f'{i}'][frame,joints.index(skl.parents()[j]),[0,2]]
                    # else:
                    #     x1 = positions_3d[f'{i}'][frame,j,[0,2]]
                    #     x2 = positions_3d[f'{i}'][frame,skl.parents()[j],[0,2]]
                    # import IPython;IPython.embed();exit()
                    cv2.line(img, pt1=tuple(x1), pt2=tuple(x2), color=color[f'{i}'],
                            thickness=1, lineType=cv2.LINE_AA, shift=0)
        writer.write(img)
    writer.release()

    if audiopath:
        outputpath = str(Path(output).parent) + '/' + str(Path(output).stem) + f'_audio_{parts}.mp4'
        instream_v = ffmpeg.input(output)
        instream_a = ffmpeg.input(audiopath)
        # import IPython;IPython.embed();exit()
        stream = ffmpeg.output(instream_v, instream_a, outputpath, vcodec='copy', acodec='aac')
        ffmpeg.run(stream)


def render_animation(data, skeleton, fps, output='interactive', bitrate=1000):
    """
    Render or show an animation. The supported output modes are:
    -- 'interactive': display an interactive figure
                       (also works on notebooks if associated with %matplotlib inline)
     -- 'html': render the animation as HTML5 video. Can be displayed in a notebook using HTML(...).
     -- 'filename.mp4': render and export the animation as an h264 video (requires ffmpeg).
     -- 'filename.gif': render and export the animation a gif file (requires imagemagick).
    """
    x = 0
    y = 1
    z = 2
    radius = torch.max(skeleton.offsets()).item() * 5 # Heuristic that works well with many skeletons

    skeleton_parents = skeleton.parents()

    plt.ioff()
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    #ax.view_init(elev=20., azim=58)
    ax.view_init(elev=90, azim=-90)

    ax.set_xlim3d([-radius/2, radius/2])
    ax.set_zlim3d([0, radius])
    ax.set_ylim3d([-radius/2, radius/2])
    # #ax.set_aspect('equal')
    # ax.set_xticklabels([])
    # ax.set_yticklabels([])
    # ax.set_zticklabels([])
    ax.dist = 7.5

    lines = []
    initialized = False

    trajectory = data[:, 0, [0, 2]]
    avg_segment_length = np.mean(np.linalg.norm(np.diff(trajectory, axis=0), axis=1)) + 1e-3
    draw_offset = int(25/avg_segment_length)
    spline_line, = ax.plot(*trajectory.T)
    camera_pos = trajectory
    height_offset = np.min(data[:, :, 1]) # Min height
    data = data.copy()
    data[:, :, 1] -= height_offset

    def update(frame):
        nonlocal initialized
        ax.set_title(frame)
        ax.set_xlim3d([-radius/2 + camera_pos[frame, 0], radius/2 + camera_pos[frame, 0]])
        ax.set_ylim3d([-radius/2 + camera_pos[frame, 1], radius/2 + camera_pos[frame, 1]])

        positions_world = data[frame]
        for i in range(positions_world.shape[0]):
            if skeleton_parents[i] == -1:
                continue
            if not initialized:
                #col = 'red' if i in skeleton.joints_right() else 'black' # As in audio cables :)
                col = 'blue'
                lines.append(ax.plot([positions_world[i, x], positions_world[skeleton_parents[i], x]],
                        [positions_world[i, y], positions_world[skeleton_parents[i], y]],
                        [positions_world[i, z], positions_world[skeleton_parents[i], z]], zdir='y', c=col))
            else:
                lines[i-1][0].set_xdata([positions_world[i, x], positions_world[skeleton_parents[i], x]])
                lines[i-1][0].set_ydata([positions_world[i, y], positions_world[skeleton_parents[i], y]])
                lines[i-1][0].set_3d_properties([positions_world[i, z], positions_world[skeleton_parents[i], z]], zdir='y')
        l = max(frame-draw_offset, 0)
        r = min(frame+draw_offset, trajectory.shape[0])
        spline_line.set_xdata(trajectory[l:r, 0])
        spline_line.set_ydata(np.zeros_like(trajectory[l:r, 0]))
        spline_line.set_3d_properties(trajectory[l:r, 1], zdir='y')
        initialized = True
        if output == 'interactive' and frame == data.shape[0] - 1:
            plt.close('all')

    #fig.tight_layout()
    anim = FuncAnimation(fig, update, frames=np.arange(0, data.shape[0]), interval=1000/fps, repeat=False)
    if output == 'interactive':
        plt.show()
        return anim
    elif output == 'html':
        return anim.to_html5_video()
    elif output.endswith('.mp4'):
        Writer = writers['ffmpeg']
        writer = Writer(fps=fps, metadata={}, bitrate=bitrate)
        anim.save(output, writer=writer)
    elif output.endswith('.gif'):
        anim.save(output, dpi=80, writer='imagemagick')
    else:
        raise ValueError('Unsupported output format (only html, .mp4, and .gif are supported)')
    plt.close()