import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def movie(data: np.array, labels, saveFileName = None):

    fig = plt.figure()

    img = plt.imshow(data[0])
    img.norm.vmin = np.min(data)
    img.norm.vmax = np.max(data)
    axes = fig.get_axes()

    labelsString = ", ".join(str(labels))
    axes[0].set_title(labelsString)

    def animate(frameNr):
        frame = data[frameNr]
        img.set_data(frame)
        plt.xlabel("Frame {},  maxval {}".format(frameNr, np.max(frame)))
        return img, labelsString

    animation = FuncAnimation(fig, animate, frames=range(data.shape[0]), interval=500, repeat=True, repeat_delay=1000)

    if not saveFileName: 
        plt.show()
    else:
        animation.save(saveFileName + ".mp4", writer='ffmpeg')
