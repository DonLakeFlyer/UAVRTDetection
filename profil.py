import cProfile
from uavrt_detection import uavrt_detection

cProfile.run("uavrt_detection()", sort="tottime")