import numpy as np

class CurrentGenerator:
    def __init__(self, seed=777, time=5000, tau=3.0, i_e0=0.5, sigmaMax=0.325,
                 sigmaMin=0.215, frequency=0.2, dt=0.025, voltage=[],
                 threshold=0.0, sigmaOpt=0.51, optimize_flag=False, simulator=None):

        self.simulator = simulator
        self.seed = np.random.seed(seed)
        self.time = time
        self.tau = tau
        self.i_e0 = i_e0
        self.i_e = []
        self.m_ie = []
        self.sigmaMax = sigmaMax
        self.sigmaMin = sigmaMin
        self.sigma = (self.sigmaMax + self.sigmaMin) / 2
        self.delta_sigma = (self.sigmaMax - self.sigmaMin) / (2 * self.sigma)
        self.sin_variance = []
        self.duration = 0.0
        self.frequency = frequency
        self.dt = dt
        self.voltage = voltage
        self.tolerance = 0.2
        self.spks = []
        self.spks_flag = False
        self.threshold = threshold
        self.optsigma = sigmaOpt
        self.spks = []
        self.optimize_flag = optimize_flag

    def generate_current(self):
        self.i_e = 0.0
        if self.optimize_flag:
            self.sin_variance = \
                [variance for variance in self.current_variance_opt()]
        else:
            self.sin_variance = \
                [variance for variance in self.current_variance()]

        ou_process = [ou for ou in self.ou_process()]

        moving_current = [mc for mc in self.moving_current_gen()]

        for n, k in enumerate(np.arange(self.time / self.dt)):
            """
            sin_sigma = sigma*(1+delta_sigma*np.sin(2*np.pi*f*t*10**-3))
            I = OU_process*sin_sigma + mu
            """

            yield self.i_e
            self.i_e = ou_process[n] * self.sin_variance[n] + moving_current[n]

    def moving_current_gen(self):
        for _ in np.arange(self.time/self.dt):
            yield self.i_e0*(1+np.sin(2 * np.pi * _
                                      * self.frequency
                                      * self.dt
                                      * 10 ** -3
                                      * 0.1
                                      ))

    def ou_process(self):
        for _ in np.arange(self.time/self.dt):
            yield self.i_e
            self.i_e = (self.i_e +
                        (self.i_e0 - self.i_e) * (self.dt / self.tau) +
                        0.5 * np.sqrt(2.0 * self.dt / self.tau) *
                        np.random.normal())

    def sub_threshold_var(self):
        selection = self.get_far_from_spikes()
        sv = np.var(self.voltage[selection])
        assert (np.max(self.voltage[selection]) < self.threshold)

        return sv

    def current_variance(self):
        for time in np.arange(self.time / self.dt):
            yield (self.sigma * (0.3 + self.delta_sigma * np.sin(2 * np.pi *
                                                                time
                                                  * self.frequency
                                                  * self.dt
                                                  * 10 ** -3)))

    def current_variance_opt(self):
        for time in np.arange(int(self.time / self.dt)):
            yield (self.optsigma * (1.0 + 0.5 * np.sin(2 * np.pi * time
                                                     * self.frequency
                                                     * self.dt
                                                     * 10 ** -3)))

    def detect_spikes(self, ref=7.0):

        """
        Detect action potentials by threshold crossing (parameter threshold,
        mV) from below (i.e. with dV/dt>0).
        To avoid multiple detection of same spike due to noise, use an
        'absolute refractory period' ref, in ms.
        """

        ref_ind = int(ref / self.dt)
        t = 0
        while t < len(self.voltage) - 1:

            if (self.voltage[t] >= self.threshold >=
                    self.voltage[t - 1]):
                self.spks.append(t)
                t += ref_ind
            t += 1

        self.spks = np.array(self.spks)
        self.spks_flag = True
        return self.spks

    def get_far_from_spikes(self, d_t_before=5.0, d_t_after=5.0):

        """
        Return indices of the trace which are in ROI. Exclude all datapoints
        which are close to a spike.
        d_t_before: ms
        d_t_after: ms
        These two parameters define the region to cut around each spike.
        """
        if not self.spks_flag:
            self.detect_spikes()

        L = len(self.voltage)
        LR_flag = np.ones(L)

        LR_flag[:] = 0.0

        # Remove region around spikes
        DT_before_i = int(d_t_before / self.dt)
        DT_after_i = int(d_t_after / self.dt)

        for s in self.spks:
            lb = max(0, s - DT_before_i)
            ub = min(L, s + DT_after_i)

            LR_flag[lb: ub] = 1

        indices = np.where(LR_flag == 0)[0]

        return indices
