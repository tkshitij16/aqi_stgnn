import numpy as np

EPA_AQI_BOUNDS = [0,50,100,150,200,300,400,500]

def aqi_to_class(aqi):
    if np.isnan(aqi): return -1
    for i in range(len(EPA_AQI_BOUNDS)-1):
        if EPA_AQI_BOUNDS[i] <= aqi <= EPA_AQI_BOUNDS[i+1]:
            return i
    return -1
