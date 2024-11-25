import torch
import matplotlib.pyplot  as plt
import matplotlib.animation as animation
import numpy as np
import cv2


def sort_seq(inputs, seq_len):
    len_sorted, sorted_idx = seq_len.sort(descending=True)
    return inputs[sorted_idx], len_sorted


def render_bones(keyps, parents, ax):
    link = [[i, parents[i]] for i in range(len(keyps))][1:]
    for l in link:
        ind0, ind1 = l[0], l[1]
        ax.plot([keyps[ind0,0], keyps[ind1,0]], [keyps[ind0,1], keyps[ind1,1]], [keyps[ind0,2], keyps[ind1,2]]) 


def save_video(keyps, save_pth):

    parents = [-1, 0, 1, 2, 3, 4, 3, 6, 7, 3, 9, 10, 0, 0]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')

    ax.set_xlim(-100,100)
    ax.set_ylim(-100,100)
    ax.set_zlim(0,200)
    ax.view_init(elev=10, azim=30)

    def update(frame):
        print("\r",f'{frame}/{len(keyps)}', end="")
        ax.cla()
        ax.set_xlim(-100, 100) 
        ax.set_ylim(-100, 100) 
        ax.set_zlim(0, 200) 
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        render_bones(keyps[frame], parents, ax)

    ani = animation.FuncAnimation(fig, update, frames=range(len(keyps)), interval = 1000/30, repeat=False)
    #render_bones(positions_3d[1000], parents, ax)
    ani.save(save_pth, writer="ffmpeg")
    #plt.show()



