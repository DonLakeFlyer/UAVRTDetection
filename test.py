import matlab as ml
import numpy as np

from Pulse import Pulse

pList = np.array([ Pulse(), Pulse() ])
pList[1].det_dec = False