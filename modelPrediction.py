import os
import tensorflow.keras as k
import datetime as dt
from radarData import getRadarData


thisDir = os.path.dirname(os.path.abspath(__file__))
tfDataDir = thisDir + "/tfData/"
modelName = "simpleRadPredModel_checkpoint.h5"


model = k.models.load_model(tfDataDir + modelName)
data = getRadarData(dt.datetime(2016, 6, 12), dt.datetime(2016, 6, 13))
prediction = model.predict(data)

print(prediction)




