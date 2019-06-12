import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def movie(data, labels, saveFileName = None):

    fig = plt.figure()

    dataToDraw = []
    for i in range(data.shape[0]):
        dataToDraw.append(data[i,:,:,0])

    img = plt.imshow(dataToDraw[0])
    axes = fig.get_axes()

    labelsString = ", ".join(str(labels))
    axes[0].set_title(labelsString)

    def animate(frame):
        img.set_data(frame)
        plt.xlabel(" maxval: " + str(np.max(frame)))
        return img, labelsString

    animation = FuncAnimation(fig, animate, interval=500, repeat=True, frames=dataToDraw)

    if not saveFileName: 
        plt.show()
    else:
        animation.save(saveFileName + ".mp4", writer='ffmpeg')
