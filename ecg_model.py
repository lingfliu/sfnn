
ECG_CHANNEL = {'I', 'II', 'III', 'aVF', 'aVL', 'aVR', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'}
FIDUCIAL = {'Ps', 'P', 'Pe', 'Q', 'R', 'S', 'Ts', 'T', 'Te', 'U'}

class Ecg:
    def __init__(self, data, channel, sps=500):
        self.data = data
        self.channel = channel
        self.sps = sps

class MultiEcg:
    def __init__(self, ecgList):
        self.ecgList = ecgList
