import radarData as rd
import numpy as np
import datetime as dt
import h5py as h5



with h5.File("processedData/test2.hdf5", "r") as f:
    for groupName in f.keys():
        group = f[groupName]
        fromTime = dt.datetime.fromtimestamp(group.attrs["fromTime"])
        lowerLeft = group.attrs["lowerLeft"]
        pixelSize = group.attrs["pixelSize"]
        frames = []
        for dsetName in group.keys():
            dset = group[dsetName]
            time = dt.datetime.fromtimestamp(dset.attrs["time"])
            data = np.array(dset)
            labels = dset.attrs["labels"]
            frame = rd.RadarFrame(time, data, lowerLeft, pixelSize)
            frame.labels = labels
            frames.append(frame)
        storm = rd.Storm(frames)
        data = [f.data for f in storm.frames]
        if np.max(data) < 150:
            rd.analyseStorm(storm)
