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
# from motionsynth_code.motion import bvh
def part2scale(parts):
    if parts == 'left_hand':
        return 1
    elif parts == 'right_hand':
        return 1
    elif parts == 'both_hands':
        return 1
    elif parts == 'full_body':
        return 5
    elif parts == 'woboth_hands':
        return 5
    elif parts == 'left_hand2':
        return 1
    else:
        raise Exception("config['parts'] has wrong name.\n please set as follows [left_hand, both_hands, full_body, woboth_hands]")
def color4(cnt):
    if cnt ==0:
        return (0,0,0)
    elif(cnt==1):
        return (256,256,0)
    elif(cnt==2):
        return (256,0,256)
    elif(cnt==3):
        return (0,256,256)
def color_joint_cv2(data, skeleton, joint,fps, output, audiopath=None,parts=None):
    #anim = bvh.load(bvhfile)[0]
    scale = part2scale(parts)
    skl = skeleton
    positions_3d = data
    radius = torch.max(skl.offsets()).item() * scale
    
    frame_total = len(positions_3d)
    fps = fps
    W = 1000
    H = 1000
    imgsize = (W, H)
    filepath = output
    codec = cv2.VideoWriter_fourcc('m','p','4','v')
    writer = cv2.VideoWriter(filepath, codec, fps, imgsize)
    positions_3d[:,:,2] = -1 * positions_3d[:,:,2] + radius
    positions_3d = positions_3d * 250/radius
    positions_3d=np.round(positions_3d).astype(int)
    root_offset = positions_3d[0,0,:]
    delta_y = 500 - root_offset[2]
    delta_x = 500 - root_offset[0]
    positions_3d[:,:,2] += delta_y
    positions_3d[:,:,0] += delta_x
    # print(skl.parents())
    for frame in range(frame_total):
        img = np.full((1000, 1000, 3), 255, dtype=np.uint8)
        
        k=0
        for i in range(positions_3d.shape[1]):
            if skl.parents()[joint[i]] in joint:
                x1 = positions_3d[frame,i,[0,2]]
                x2 = positions_3d[frame,joint.index(skl.parents()[joint[i]]),[0,2]]
                # print(x1,x2)
                cv2.line(img, pt1=tuple(x1), pt2=tuple(x2), color=color4(k%4),
                        thickness=1, lineType=cv2.LINE_AA, shift=0)
                inter_x = (x1[0]+x2[0])/2
                inter_y = (x1[1]+x2[1])/2
                
                # import IPython;IPython.embed();exit()
                cv2.putText(img, f'({i} to {skl.parents()[i]})', (int(inter_x),int(inter_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color4(k%4), 1, cv2.LINE_AA)
                k+=1
        writer.write(img)
    writer.release()

    if audiopath:
        outputpath = str(Path(output).parent) + '/' + str(Path(output).stem) + f'{parts}_audio.mp4'
        instream_v = ffmpeg.input(output)
        instream_a = ffmpeg.input(audiopath)
        stream = ffmpeg.output(instream_v, instream_a, outputpath, vcodec='copy', acodec='aac')
        ffmpeg.run(stream)


def render_joint_cv2(data, skeleton, fps, output, audiopath=None):
    #anim = bvh.load(bvhfile)[0]
    skl = skeleton
    positions_3d = data
    radius = torch.max(skl.offsets()).item() * 5

    #positions_3d[:,:,0] *= -1

    #normalize

    frame_total = len(positions_3d)
    fps = fps
    W = 1000
    H = 1000
    imgsize = (W, H)
    filepath = output
    codec = cv2.VideoWriter_fourcc('m','p','4','v')
    writer = cv2.VideoWriter(filepath, codec, fps, imgsize)
    positions_3d[:,:,2] = -1 * positions_3d[:,:,2] + radius
    positions_3d = positions_3d * 250/radius

    positions_3d=np.round(positions_3d).astype(int)
    root_offset = positions_3d[0,0,:]
    delta_y = 500 - root_offset[2]
    delta_x = 500 - root_offset[0]
    positions_3d[:,:,2] += delta_y
    positions_3d[:,:,0] += delta_x
    print(skl.parents())
    #cv2.namedWindow('window')
    for frame in range(frame_total):
        img = np.full((1000, 1000, 3), 1, dtype=np.uint8)
        # import IPython;IPython.embed();exit()
        for i in range(positions_3d.shape[1]):
            if skl.parents()[i] >= 0:
                x1 = positions_3d[frame,i,[0,2]]
                x2 = positions_3d[frame,skl.parents()[i],[0,2]]
                # import IPython;IPython.embed();exit()
                # print(x1,x2)
                cv2.line(img, pt1=tuple(x1), pt2=tuple(x2), color=(0,255,0),
                        thickness=1, lineType=cv2.LINE_AA, shift=0)
        writer.write(img)
    writer.release()

    if audiopath:
        outputpath = str(Path(output).parent) + '/' + str(Path(output).stem) + '_audio.mp4'
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