import numpy as np
from scipy.integrate import solve_ivp
from glv_functions import f, event

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
        # This function updates the final abundances, rows are the samples and columns represent the species.
        try:
            assert isinstance(perturbation, bool)
        except AssertionError:
            raise ValueError('perturbation must be boolean')

        Y = self.Y_alt if perturbation else self.Y

        Final_abundances = np.zeros((self.n, self.smp))
        Final_abundances_single_sample = np.zeros(self.n)

        if self.smp > 1:  # Solution for cohort.
            for m in range(self.smp):

                event_idx = None

                t_temp = 0

                while event_idx is None:

                    # Set the parameters to the functions f and event.
                    f_with_params = lambda t, x: f(t, x, self.r.view(), self.s.view(), self.A.view(), self.delta)
                    event_with_params = lambda t, x: event(t, x, self.r.view(), self.s.view(),
                                                           self.A.view(), self.delta)

                    # solve GLV.
                    sol = solve_ivp(f_with_params, (0 + t_temp, self.final_time + t_temp),
                                    Y[m, :], max_step=self.max_step, events=event_with_params)

                    # Check if the steady state was reached.
                    if len(sol.t_events[0]) > 0:
                        event_time = sol.t_events[0][0]
                        event_idx = np.argmin(np.abs(sol.t - event_time))
                        Final_abundances[:, m] = sol.y[:, event_idx]
                    else:
                         Y[:][m] = sol.y[:, -1]
                         t_temp = sol.t[-1]

            # Normalize the results.
            Final_abundances = self._normalize_results(Final_abundances)
            return Final_abundances

        else:  # Solution for single sample.

            event_idx = None

            t_temp = 0

            while event_idx is None:

                # Set the parameters to the functions f and event.
                f_with_params = lambda t, x: f(t, x, self.r.view(), self.s.view(), self.A.view(), self.delta)
                event_with_params = lambda t, x: event(t, x, self.r.view(), self.s.view(),
                                                       self.A.view(), self.delta)

                # solve GLV.
                sol = solve_ivp(f_with_params, (0 + t_temp, self.final_time + t_temp),
                                Y[:], max_step=self.max_step, events=event_with_params)

                # Check if the steady state was reached.
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
        # Normalization of the final abundances.
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
        # Create interaction matrix.
        interaction_matrix = np.zeros((self.n, self.n))
        random_values = np.random.uniform(0, 1, (self.n, self.n))
        mask = random_values < self.p_mat
        interaction_matrix[mask] = np.random.normal(0, self.sigma, size=mask.sum())
        return interaction_matrix

    def _create_noisy_interaction_matrix(self):
        pass

    def _create_set_of_initial_conditions(self):
        # Create initial conditions for the cohort.
        if self.smp != 1:
            init_cond_set = np.zeros((self.smp, self.n))
            random_values = np.random.uniform(0, 1, (self.smp, self.n))
            mask = random_values < self.p_init
            init_cond_set[mask] = random_values[mask]
        # Create initial conditions for single sample.
        else:
            init_cond_set = np.zeros(self.n)
            random_values = np.random.uniform(0, 1, self.n)
            mask = random_values < self.p_init
            init_cond_set[mask] = random_values[mask]

        return init_cond_set

    def _create_r_vector(self):
        r = np.random.uniform(0, 1, self.n)
        return r

    def _create_s_vector(self):
        s = np.ones(self.n)
        return s

    def _create_alt_set_of_initial_conditions(self):
        # Creation of the alternative set of initial conditions, by species abundance of the current initial
        #condition to zero with probability p_alt_init.
        alt_init_cond_set = self.Y.copy()

        non_zero_indices = np.nonzero(alt_init_cond_set)
        random_values = np.random.uniform(0, 1, size=non_zero_indices[0].shape)
        alt_init_cond_set[non_zero_indices] = np.where(random_values < self.p_alt_init, 0,
                                                       alt_init_cond_set[non_zero_indices])

        return alt_init_cond_set
