# Import the necessary packages
import math
from typing import List, Tuple
import numpy as np
import pandas as pd
np.random.seed(1234)
from numpy.typing import ArrayLike
from hmm import HMM  # class from hmm.py, which implements the inference methods from the previous tutorials

class HMMOptimiser(object):
    """
    Implements the Baum-Welch (EM) algorithm to learn the parameters of a discrete HMM model.

    Attributes:
        model:             an HMM model from hmm.py
        num_hiddens:       an integer indicating the number of possible hidden states
        num_observations:  an integer indicating the number of possible observed states
    """
    def __init__(self,
                num_hiddens:int,
                num_observations:int) -> None:
        super().__init__()
        assert num_hiddens is not None and num_observations is not None
        self.model = None
        self.num_hiddens = num_hiddens
        self.num_observations = num_observations

    def _e_step(self, data_loader: object) -> Tuple[float, List, List]:
        hk_list = []
        hkk_list = []
        log_ps = []

        for o_seq in data_loader:
            _, _, log_p, hk, hkk = self.model.marginal(o_seq)

            hk_list.append(hk)
            hkk_list.append(hkk)
            log_ps.append(log_p)

        return np.mean(log_ps), hk_list, hkk_list

    def _m_step(self,
                data_loader: object,
                hk_list: List,
                hkk_list: List
                ) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        _initial_ = np.zeros(self.num_hiddens)
        _transition_ = np.zeros([self.num_hiddens, self.num_hiddens])
        _emission_ = np.zeros([self.num_observations, self.num_hiddens])

        for j, obs in enumerate(data_loader):
            # Retrieve the distributions inferred in the E-step for the current observation obs
            hk = hk_list[j]
            hkk = hkk_list[j]
            # Handle obs of length 1 for which hkk is None
            hkk = hkk if hkk is not None else float('-inf') * np.ones([1, self.num_hiddens, self.num_hiddens])

            _initial_ += np.exp(hk[0])  # your code here
            _transition_ += np.exp(hkk).sum(axis=0)  # your code here
            for m, ob in enumerate(obs):
                _emission_[ob] += np.exp(hk[m])  # your code here
                # hint: _emission_[ob] += np.exp(hk) would cause an error

        # Normalise the distributions
        _initial_ /= len(data_loader)
        _transition_ /= _transition_.sum(axis=1, keepdims=True)
        _emission_ /= _emission_.sum(axis=0, keepdims=True)

        return _initial_, _transition_, _emission_

    def _initial_params(self) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        initial = np.random.uniform(size=self.num_hiddens)
        initial /= initial.sum(axis=0)

        transition = np.random.uniform(size=[self.num_hiddens, self.num_hiddens])
        transition /= transition.sum(axis=1, keepdims=True)

        emission = np.random.uniform(size=[self.num_observations, self.num_hiddens])
        emission /= emission.sum(axis=0, keepdims=True)

        return initial, transition, emission

    @staticmethod
    def _stop_criterion(step: int = 0,
                        delta_param: float = 1e-3,
                        delta_logpx: float = 1e-1,
                        ) -> bool:
        max_steps = 100
        min_delta_param = 1e-16
        min_delta_logpx = 1e-8
        stop_condition = (step >= max_steps
                          or delta_param < min_delta_param
                          or delta_logpx < min_delta_logpx)  # your code here
        return stop_condition

    def baum_welch(self, data_loader: List[List[int]], verbose: bool = True):
        # Step 1: initialise the parameters for HMM model
        initial, transition, emission = self._initial_params()
        self.model = HMM(np.log(initial),
                         np.log(transition),
                         np.log(emission)
                         )

        # Step 2: set up the following variables for repeating the E/M-steps.
        stop = False  # flag for stopping the loop
        step = 0  # track the number of steps
        delta_param = math.inf  # track the change of parameters
        delta_loglikelihood = math.inf  # track the change of log-likelihood
        last_loglikelihood = 0.
        loglikelihood_list = []

        # Step 3: repeat the E/M-steps
        while not stop:
            # step 3.1: E-step
            loglikelihood, hk_list, hkk_list = self._e_step(data_loader)
            # step 3.2: M-step
            _initial_, _transition_, _emission_ = self._m_step(data_loader, hk_list, hkk_list)

            # step 3.3: track step and change of parameters/log-likelihoods
            step += 1
            delta_param = self.model.get_delta_param(
                np.log(_initial_),
                np.log(_transition_),
                np.log(_emission_)
            )
            delta_loglikelihood = abs(loglikelihood - last_loglikelihood)
            last_loglikelihood = loglikelihood

            # step 3.4: update the parameters of HMM model
            self.model.initial = np.log(_initial_)
            self.model.transition = np.log(_transition_)
            self.model.emission = np.log(_emission_)

            # step 3.5: check if we should end the loop now
            stop = self._stop_criterion(step, delta_param, delta_loglikelihood)

            # monitor the learning procedure
            loglikelihood_list.append(loglikelihood)
            if verbose:
                print('step:', step, '\tloglikelihood:', loglikelihood)

        self._trained_ = True

        return loglikelihood_list




initial = np.array([0.2, 0.8])
transition = np.array([[0.2, 0.8],
                           [0.6, 0.4]])
emission = np.array([[0.0, 0.1],  # probability of emitting <EOS>
                         [0.3, 0.8],
                         [0.7, 0.1]])


ref = pd.read_csv('data/refpanel.txt', sep='\t',engine='python')



from data import DataLoader
dataloader = DataLoader(initial=initial,
                            transition=transition,
                            emission=emission)

data_list = dataloader.get_data_list(300)

# Step 3: fit an HMM by using HMMOptimiser class
optim = HMMOptimiser(num_hiddens=2, num_observations=3)
_ = optim.baum_welch(data_list)
hmm = optim.model

# Step 4: print the parameters fit on the synthetic data
# Note that parameters of our HMM class are in the log space
print('true initial:\n', initial)
print('fitted initial:\n', np.exp(hmm.initial))
print('true transition:\n', transition)
print('fitted transition:\n', np.exp(hmm.transition))
print('true emission:\n', emission)
print('fitted emission:\n', np.exp(hmm.emission))
