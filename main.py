from Simulator import Simulator
from modelfit import GIFFit




try:
    simulator = Simulator()
    simulator.main(optimize=False, train_time=120, test_time=50)
    GIFFit(simulator=simulator, plot=True).run()

except Exception as e:
    print(e)