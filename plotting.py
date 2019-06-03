import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from radarData import getRandomBatch, translateLabels


def movie(data, labels, saveFileName = None):
    fig = plt.figure()
    dataToDraw = data[0,:,:,0]
    labelsString = ", ".join(translateLabels(labels))
    img = plt.imshow(dataToDraw)
    axes = fig.get_axes()
    axes[0].set_title(labelsString)
    def animate(frame_number):
        if frame_number >= data.shape[0]:
            animation.event_source.stop()
        else: 
            dataToDraw = data[frame_number,:,:,0]
            img.set_data(dataToDraw)
        return img, labelsString
    animation = FuncAnimation(fig, animate, interval=500, repeat=True)
    if not saveFileName: 
        plt.show()
    else:
        animation.save(saveFileName + ".mp4", writer='ffmpeg')



if __name__ == "__main__":

    batchSize = 3
    timeSteps = 10
    imageSize = 51
    data, labels = getRandomBatch(batchSize, timeSteps, imageSize)

    movie(data[0], labels[0])
