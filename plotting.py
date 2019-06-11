import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def movie(data, labels, saveFileName = None):
    fig = plt.figure()
    dataToDraw = data[0,:,:,0]
    labelsString = ", ".join(str(labels))
    img = plt.imshow(dataToDraw)
    axes = fig.get_axes()
    axes[0].set_title(labelsString)
    def animate(frame_number):
        if frame_number >= data.shape[0]:
            animation.event_source.stop()
        else: 
            dataToDraw = data[frame_number,:,:,0]
            img.set_data(dataToDraw)
            plt.xlabel("frame: " + str(frame_number))
        return img, labelsString
    animation = FuncAnimation(fig, animate, interval=500, repeat=True)
    if not saveFileName: 
        plt.show()
    else:
        animation.save(saveFileName + ".mp4", writer='ffmpeg')
