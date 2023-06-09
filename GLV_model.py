import numpy as np
from scipy.integrate import solve_ivp

class GLV:
    """
    This class is responsible to solve the GLV model with verification of reaching the steady state
    for a given parameters.
    """
    def __init__(self, n_samples, n_species, delta, final_time, max_step, p_mat, p_init, p_alt_init, sigma):
        """
        :param n_samples: The number of samples you are need to compute.
        :param n_species: The number of species at each sample.
        :param delta: This parameter is responsible for the stop condition at the steady state.
        :param final_time: the final time of the integration.
        :param max_step: maximal allowed step size.
        :param p_mat: probability of inserting element to the interaction matrix.
        :param p_init: probability of inserting species initial abundance.
        :param p_alt_init: probability of destroying species initial abundance in the alternative initial condition.
        :param sigma: the standard deviation of the interactions.
        """
        self.smp = n_samples
        self.n = n_species
        try:
            assert(self.smp > 0)
            assert(self.n > 0)
            assert isinstance(self.smp, int)
            assert isinstance(self.n, int)
        except AssertionError:
            raise ValueError("n_samples and n_species should be positive integers")
        self.p_mat = p_mat
        self.p_init = p_init
        self.p_alt_init = p_alt_init
        try:
            assert(0 <= p_mat <= 1)
            assert(0 <= p_init <= 1)
            assert(0 <= p_alt_init <= 1)
        except AssertionError:
            raise ValueError("p_mat, p_init and p_alt_init should be between 0 and 1")
        self.delta = delta
        self.sigma = sigma
        try:
            assert(delta > 0)
            assert(sigma > 0)
            assert isinstance(delta, (float, int))
            assert isinstance(sigma, (float, int))
        except AssertionError:
            raise ValueError("delta and sigma should be positive numbers")
        self.final_time = final_time
        self.max_step = max_step
        try:
            assert(final_time > 0)
            assert(max_step > 0)
            assert isinstance(final_time, (float, int))
            assert isinstance(max_step, (float, int))
            assert(final_time > max_step)
        except AssertionError:
            raise ValueError("final_time and max_step should be positive numbers and final_time > max_step")
        self.r = self._create_r_vector()
        self.s = self._create_s_vector()
        self.A = self._create_interaction_matrix()
        self.Y = self._create_set_of_initial_conditions()
        self.Y_alt = self._create_alt_set_of_initial_conditions()

    def solve(self, perturbation=False):
        """
        This function updates the final abundances, rows are the species and columns represent the samples.
        """
        try:
            assert isinstance(perturbation, bool)
        except AssertionError:
            raise ValueError('perturbation must be boolean')
        if perturbation:
            Y = self.Y_alt
        else:
            Y = self.Y
        def f(t, x):
            """
            GLV formula.
            """
            return np.array([self.r[i] * x[i] - self.s[i] * x[i] ** 2 + sum([self.A[i, p] * x[
                i] * x[p] for p in range(self.n) if p != i]) for i in range(self.n)])

        def event(t, x):
            """
            Event function that triggers when dxdt is close to steady state.
            """
            return max(abs(f(t, x))) - self.delta

        Final_abundances = np.zeros((self.n, self.smp))
        Final_abundances_single_sample = np.zeros(self.n)

        if self.smp > 1:  # Solution for cohort.
            for m in range(self.smp):
                print(m)
                # Get the index at which the event occurred.
                event_idx = None

                t_temp = 0

                while event_idx is None:

                    # solve GLV up to time span.
                    sol = solve_ivp(f, (0 + t_temp, self.final_time + t_temp),
                                    Y[:][m], max_step=self.max_step, events=event)

                    if len(sol.t_events[0]) > 0:
                        event_time = sol.t_events[0][0]
                        event_idx = np.argmin(np.abs(sol.t - event_time))
                        Final_abundances[:, m] = sol.y[:, event_idx]
                    else:
                         Y[:][m] = sol.y[:, -1]
                         t_temp = sol.t[-1]

            Final_abundances = self._normalize_results(Final_abundances)
            return Final_abundances

        else:  # Solution for single sample.

            event_idx = None

            t_temp = 0

            while event_idx is None:

                sol = solve_ivp(f, (0 + t_temp, self.final_time + t_temp),
                                Y[:], max_step=self.max_step, events=event)

                if len(sol.t_events[0]) > 0:
                    event_time = sol.t_events[0][0]
                    event_idx = np.argmin(np.abs(sol.t - event_time))
                    Final_abundances_single_sample[:] = sol.y[:, event_idx]
                else:
                    Y[:] = sol.y[:, -1]
                    t_temp = sol.t[-1]

                # Save the solution up to the event time
            Final_abundances_single_sample = self._normalize_results(Final_abundances_single_sample)
        return Final_abundances_single_sample

    def _normalize_results(self, final_abundances):
        """
        Normalization of the final abundances.
        """
        if self.smp > 1:  # Normalization for cohort
            norm_factors = np.sum(final_abundances, axis=0)
            final_abundances_norm = np.array([final_abundances[:, i] / norm_factors[i] for i in range(
                0, np.size(norm_factors))])
            return final_abundances_norm
        else:  # Normalization for single sample
            norm_factor = np.sum(final_abundances)
            final_abundances_norm = final_abundances/norm_factor
            return final_abundances_norm

    def _create_interaction_matrix(self):
        interaction_matrix = np.zeros([self.n, self.n])
        for row, col in np.ndindex(interaction_matrix.shape):
            if np.random.uniform(0, 1) < self.p_mat:
                interaction_matrix[row, col] = np.random.normal(0, self.sigma)
            else:
                interaction_matrix[row, col] = 0
        return interaction_matrix
    def _create_noisy_interaction_matrix(self):
        pass

    def _create_set_of_initial_conditions(self):
        if self.smp != 1:
            init_cond_set = np.zeros([self.smp, self.n])
            for y in init_cond_set:
                for i in range(0, self.n):
                    if np.random.uniform(0, 1) < self.p_init:
                        y[i] = np.random.uniform(0, 1)
                    else:
                        y[i] = 0
        else:
            init_cond_set = np.zeros(self.n)
            for i in range(0, self.n):
                if np.random.uniform(0, 1) < self.p_init:
                    init_cond_set[i] = np.random.uniform(0, 1)
                else:
                    init_cond_set[i] = 0
        return init_cond_set

    def _create_r_vector(self):
        r = np.random.uniform(0, 1, self.n)
        return r

    def _create_s_vector(self):
        s = np.ones(self.n)
        return s

    def _create_alt_set_of_initial_conditions(self):
        alt_init_cond_set = self.Y.copy()
        for y in alt_init_cond_set:
            for i in range(0, self.n):
                if y[i] != 0:
                    if np.random.uniform(0, 1) < self.p_alt_init:
                        y[i] = 0
        return alt_init_cond_set

    def solve_alt(self):
        pass
