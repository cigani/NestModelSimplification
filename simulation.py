from __future__ import print_function
import os
import sys
from bisect import bisect_left

import numpy as np

import current_generation

import cPickle as pickle
import neuron

from matplotlib import pyplot as plt
import seaborn as sns

import os

def data_records(dictionaryofvalues, path):
    import h5py
    import time
    timestr = time.strftime("%m.%d.%H:%M:%S")

    #Create path if it is not found
    if not os.path.isdir(path):
        os.makedirs(path)

    #Where the training and test data is going to be stored
    H5_OUTPUT_PATH = '{0}/data.{1}.{2}'.format(path, timestr, 'hdf5')

    print("Saving to: {0}".format(H5_OUTPUT_PATH))

    data_file = h5py.File('{}'.format(H5_OUTPUT_PATH), 'w')
    for keys, values in dictionaryofvalues.iteritems():
        saved = data_file.create_dataset('{0}'.format(keys),
                                         data=np.array(values),
                                         compression='gzip')
    data_file.close()


def data_print_static(data):
    """

    :rtype: Prints one line to Terminal
    """
    sys.stdout.write("\r\x1b[K" + data)
    sys.stdout.flush()


def take_closest(my_list, my_number):
    """
    Assumes my_list is sorted. Returns closest value to my_number.

    If two numbers are equally close, return the smallest number.
    """
    pos = bisect_left(my_list, my_number)
    if pos == 0:
        return my_list[0]
    if pos == len(my_list):
        return my_list[-1]
    before = my_list[pos - 1]
    after = my_list[pos]
    print("Before: {0}. Before Pos-1: {1}".format(before, pos - 1))
    print("After: {0}. After Pos: {1}".format(after, pos))
    if after - my_number < my_number - before:
        return pos
    else:
        return pos - 1


def find_opt(input_list, val):
    return min(range(len(input_list)), key=lambda i: abs(input_list[i] - val))


def init_simulation():
    """Initialise simulation environment"""

    neuron.h.load_file("stdrun.hoc")
    neuron.h.load_file("import3d.hoc")

    print('Loading constants')
    neuron.h.load_file('constants.hoc')


class Simulator:
    def __init__(self, **kwargs):

        #Setting of paths
        self.MODEL_PATH = kwargs.get('model_path','/Users/vlasteli/Documents/Models/L5_TTPC1_cADpyr232_1')
        if not os.path.isdir(self.MODEL_PATH):
            os.makedirs(self.MODEL_PATH)
        self.SIMULATION_PATH = kwargs.get('sim_path', os.path.join(self.MODEL_PATH, 'simulation'))
        if not os.path.isdir(self.SIMULATION_PATH):
            os.makedirs(self.SIMULATION_PATH)
        self.PARAMETERS_PATH = kwargs.get('parameters_path', os.path.join(self.SIMULATION_PATH, 'parameters'))
        if not os.path.isdir(self.PARAMETERS_PATH):
            os.makedirs(self.PARAMETERS_PATH)

        self.CELL_TEMPLATE = kwargs.get('cell_template_name', 'cADpyr232_L5_TTPC1_0fb1ca4724')


        #Figures path
        self.FIGURES_PATH = kwargs.get('figures_path', os.path.join(self.SIMULATION_PATH, 'figures'))
        if not os.path.isdir(self.FIGURES_PATH):
            os.makedirs(self.FIGURES_PATH)

        #Add model path as to HOC_LIBRARY_PATH
        os.environ['HOC_LIBRARY_PATH'] = os.path.join(self.MODEL_PATH)

        # Creation Variables
        self.currentFlag = False
        self.recordings = []
        self.stimuli = []
        self.cell = []

        """
                :param time: simulation time
                :param sigmamax: sigmaMax used to determine Sigma and DeltaSigma
                :param sigmamin: sigmaMin used to determine Sigma and DeltaSigma
                :param i_e0: Injected current without noise
        """

        self.time = kwargs.get('time',3000.0)
        self.sigmamax = kwargs.get('sigmamax', 0.325)
        self.sigmamin = kwargs.get('sigmamin', 0.215)
        self.i_e0 = kwargs.get('i_e0', 0.16)
        self.dt = kwargs.get('dt', 0.025)
        self.plot = kwargs.get('plot', False)



        # Injection current
        self.playVector = []
        self.current = []

        # Recorded values
        self.rvoltage = []
        self.rtime = []
        self.rcurrent = []

        # Optimization
        self.optimize = kwargs.get('optimize', False)
        self.sigmaopt = kwargs.get('sigmaopt', 0.15)
        self.variance = []
        self.varPlot = []
        self.sigmaoptPlot = []
        self.deltasigma = kwargs.get('deltasigma', 0.005)
        self.spks = []
        self.hz = kwargs.get('hz',0.0)
        self.RandomSeed = 777

        # Current generating class
        self.cg = current_generation.CurrentGenerator

    def plotcurrent(self, val):
        plt.clf()
        plt.plot(val)
        plt.xlabel('time (ms)')
        plt.ylabel('I (nA)')
        print('Saveing figure: ', os.path.join(self.FIGURES_PATH, 'current.eps'))
        plt.savefig(os.path.join(self.FIGURES_PATH, 'current.eps'))
        if self.plot:
            plt.show()

    def create_cell(self, add_synapses=True):
        # Load morphology
        """
        Creates the cell in Neuron
        :return: Cell
        :rtype: Hoc
        """
        neuron.h.load_file(os.path.join("morphology.hoc"))
        # Load biophysics
        neuron.h.load_file("biophysics.hoc")
        # Load main cell template
        neuron.h.load_file("template.hoc")

        # Instantiate the cell from the template
        self.cell = getattr(neuron.h, self.CELL_TEMPLATE)(1 if add_synapses
                                                           else 0)

        return self.cell

    def create_stimuli(self):
        """
        Create stimulus input
        :return: Current Clamp
        :rtype: Neuron <HOC> Object
        """
        self.stimuli = neuron.h.IClamp(0.5, sec=self.cell.soma[0])
        self.stimuli.delay = 0
        self.stimuli.dur = 1e9

        return self.stimuli

    def create_current(self):
        """
        Generate the noisy current needed for injection
        """
        cg = current_generation.CurrentGenerator(time=self.time, i_e0=self.i_e0,
                                                 sigmaMax=self.sigmamax,
                                                 sigmaMin=self.sigmamin,
                                                 sigmaOpt=self.sigmaopt,
                                                 seed=self.RandomSeed,
                                                 optimize_flag=False)
        self.current = [x for x in cg.generate_current()]
        self.playVector = neuron.h.Vector(np.size(self.current))

        for k in xrange(np.size(self.current)):
            self.playVector.set(k, self.current[k])

        self.currentFlag = False

    def create_recordings(self):
        """
        Generates the Dictionary and Vectors used to store Neuron data
        :return: Time, Voltage, Current
        :rtype:  Dictionary ['time', 'voltage', 'current']
        """
        self.recordings = {'time': neuron.h.Vector(),
                           'voltage': neuron.h.Vector(),
                           'current': neuron.h.Vector()}
        self.recordings['current'].record(self.stimuli._ref_amp, 0.1)
        self.recordings['time'].record(neuron.h._ref_t, 0.1)
        self.recordings['voltage'].record(self.cell.soma[0](0.5)._ref_v, 0.1)

        return self.recordings

    def record_recordings(self):
        self.rtime = np.array(self.recordings['time'])
        self.rvoltage = np.array(self.recordings['voltage'])
        self.rcurrent = np.array(self.recordings['current'])
        recordings_dir = 'python_recordings'
        soma_voltage_filename = os.path.join(
            self.SIMULATION_PATH,
            'soma_voltage_step.dat')
        np.savetxt(
            soma_voltage_filename,
            np.transpose(
                np.vstack((
                    self.rtime,
                    self.rvoltage,
                    self.rcurrent))))

    def run_step(self, time, train=True):
        self.time = time
        neuron.h.tstop = self.time
        self.create_current()
        self.playVector.play(self.stimuli._ref_amp, neuron.h.dt)
        print('Running for %f ms' % neuron.h.tstop)
        neuron.h.run()
        self.rvoltage = np.array(self.recordings['voltage'])
        self.rcurrent = np.array(self.recordings['current'])
        if not self.optimize:
            if train:
                data_records(self.recordings, os.path.join(self.SIMULATION_PATH, 'train'))
            else:
                data_records(self.recordings, os.path.join(self.SIMULATION_PATH, 'test'))

    def brute_optimize_ie(self, current_params_output=os.path.join('params', 'current_params.pck')):
        while self.hz < 3.5 or self.hz > 5.5:
            self.optmize_ie()
            self.spks = self.cg(
                voltage=self.rvoltage[1000 * 10:]).detect_spikes()
            if self.spks.size:
                self.hz = len(self.spks) / (self.time / 1000.0)
            else:
                self.hz = 0.0
            data_print_static("i_e0: {0}, Hz: {1}"
                              .format(self.i_e0,
                                      self.hz))
            if self.hz <= 3.5:
                self.i_e0 += 0.05
            elif self.hz > 5.5:
                self.i_e0 -= 0.05
        current_paras = {"i_e0": self.i_e0}
        pickle.dump(current_paras, open(
            current_params_output, "wb"))

        self.plotcurrent(self.current)

    def optmize_ie(self):
        self.time = 15000
        self.run_step(self.time)

    def run_optimize_sigma(self):
        self.optimize_play_vector()
        self.playVector.play(self.stimuli._ref_amp, neuron.h.dt)
        neuron.h.run()
        self.rvoltage = np.array(self.recordings['voltage'])
        self.variance = self.cg(voltage=self.rvoltage[
                                        1000 * 10:]).sub_threshold_var()

    def optimize_play_vector(self):
        self.time = 10000
        neuron.h.tstop = self.time
        self.i_e0 = 0.0

        # Be sure to set the flag here
        cg = current_generation.CurrentGenerator(time=self.time,
                                                 sigmaOpt=self.sigmaopt,
                                                 optimize_flag=True)

        self.current = [x for x in cg.generate_current()]
        assert (np.size(self.current) == self.time / self.dt)

        self.playVector = neuron.h.Vector(np.size(self.current))

        for k in xrange(np.size(self.current)):
            self.playVector.set(k, self.current[k])
        return self.playVector

    def brute_optimize_sigma(self, sigmas_output=os.path.join('parameters', 'sigmas_output.pck')):
        n = 1
        while self.variance < 7 or not self.variance:
            self.run_optimize_sigma()
            data_print_static("Optimizing Sigma: {0}. "
                              "Current Sigma: {1}. Current Var: {2}."
                              .format(n, self.sigmaopt, self.variance))
            print("")

            self.varPlot.append(self.variance)
            self.sigmaoptPlot.append(self.sigmaopt)
            self.sigmaopt += self.deltasigma
            n += 1

        sminIndex = find_opt(self.varPlot, 3)
        smaxIndex = find_opt(self.varPlot, 7)
        self.sigmamin = self.sigmaoptPlot[sminIndex]
        self.sigmamax = self.sigmaoptPlot[smaxIndex]
        self.plot_trace(self.rvoltage[1000 * 10:], 'rvoltage')
        if self.varPlot[sminIndex] > 4:
            raise Exception("Sigma Minimum is above acceptable range. "
                            "Initiate fitting with smaller Sigma")
        elif self.varPlot[sminIndex] < 2:
            raise Exception("Sigma Minimum is below acceptable range. "
                            "Initiate fitting with smaller d_sigma")
        if 5 > self.varPlot[smaxIndex] > 9:
            raise Exception("Sigma Maximum is out of bounds. "
                            "Initiate fitting with smaller d_sigma.")

        print("")
        print("Optimization Complete: Sigma Min: {0}. Sigma Max {1}.".format(
            self.sigmamin, self.sigmamax))

        sigmas = {"sigmamin": self.sigmamin,
                  "sigmamax": self.sigmamax}

        pickle.dump(sigmas, open(
            sigmas_output, "wb"))

    def plot_trace(self, val, name='trace', ylabel='Vm (mV)', save=False):
        plt.plot(val)
        plt.ylabel(ylabel)
        plt.xlabel('time (ms)')

        print ('Saving figure: ', os.path.join(self.FIGURES_PATH, "{0}_plot.eps".format(name)))

        if save:
            plt.savefig(os.path.join(self.FIGURES_PATH, "{0}_plot.eps".format(name)))

        if self.plot:
            plt.show()

    def load_parameters(self):
        # Load sigmas
        try:
            sigmas = pickle.load(open(os.path.join(self.PARAMETERS_PATH, 'sigmas.pck'), 'r'))
            self.sigmamin = sigmas['sigmamin']
            self.sigmamax = sigmas['sigmamax']
        except Exception as e:
            print(e, "Doing with default values...")

        # Load i_e0
        try:
            current = pickle.load(open(os.path.join(self.PARAMETERS_PATH, 'current_params.pck'), 'r'))
            self.i_e0 = current['i_e0']
        except Exception as e:
            print(e, "Doing with default values...")

    def plot_current_voltage(self):
        plt.subplot(211)
        self.plot_trace(self.recordings['voltage'], save=False)
        plt.subplot(212)
        self.plot_trace(self.recordings['current'], 'training_current_and_voltage_optimization', ylabel='I(nA)',
                        save=True)
        plt.clf()

    def main(self, optimize=False, train_time=130000, test_time=21000, test_num=5):
        """
        :param optimize:
        :param train_time:
        :param test_time:
        :param test_num:
        :return:
        """

        """
        First compile mechanisms, else they won't be found in the .hoc files,
        interestingly only relative path works for compilation.
        """
        mechanisms_relative_path = os.path.relpath(os.path.join(self.MODEL_PATH, "mechanisms"))
        os.system("nrnivmodl {0} ".format(mechanisms_relative_path))
        print('Reloading neuron...')
        reload(neuron)

        #Change working directory to model directory
        OLD_DIR = os.path.dirname(os.path.realpath(__file__))
        os.chdir(self.MODEL_PATH)
        os.system('cd ' + self.MODEL_PATH)
        #Check if mechanisms are compiled
        print(os.system('pwd'))

        self.optimize = optimize
        init_simulation()
        self.cell = self.create_cell(add_synapses=False)
        self.stimuli = self.create_stimuli()
        self.recordings = self.create_recordings()
        neuron.h.tstop = self.time
        neuron.h.cvode_active(0)
        if optimize:
            #self.brute_optimize_sigma(sigmas_output=os.path.join(self.PARAMETERS_PATH, 'sigmas.pck'))
            self.brute_optimize_ie(current_params_output=os.path.join(self.PARAMETERS_PATH, 'current_params.pck'))
            plt.subplot(211)
            self.plot_current_voltage()
        else:
            self.load_parameters()
            self.run_step(train_time, True)
            self.plot_current_voltage()

            for n in range(test_num):
                self.run_step(test_time, False)

        #Change back to working directory
        os.chdir(OLD_DIR)
