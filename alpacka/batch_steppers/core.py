"""Environment steppers."""

from alpacka import data


class BatchStepper:
    """Base class for running a batch of steppers.

    Abstracts out local/remote prediction using a Network.

    Subclasses should override _run_episode_batch() method.
    """

    def __init__(
        self, env_class, agent_class, network_fn, n_envs, output_dir
    ):
        """No-op constructor just for documentation purposes.

        Args:
            env_class (type): Environment class.
            agent_class (type): Agent class.
            network_fn (callable): Function () -> Network. Note: we take this
                instead of an already-initialized Network, because some
                BatchSteppers will send it to remote workers and it makes no
                sense to force Networks to be picklable just for this purpose.
            n_envs (int): Number of parallel environments to run.
            output_dir (str or None): Experiment output dir if the BatchStepper
                is initialized from Runner, None otherwise.
        """
        del env_class
        del agent_class
        del network_fn
        del output_dir
        self.n_envs = n_envs

    def _prepare_solve_kwargs(self, batched_solve_kwargs, common_solve_kwargs):
        """ Construct solve_kwargs for each agent.

        Args:
            batched_solve_kwargs (dict): Batched keyword arguments passed to
                Agent.solve(). This should be used if agents should get
                different parameters to solve(). If all agent are supposed to
                use the same kwargs, pass them via **common_solve_kwargs.

            common_solve_kwargs (dict): Keyword arguments passed to
                Agent.solve().

        Returns:
            List of kwargs per agent.

        Example:
            Input:
                common_solve_kwargs = dict(a=1, b=2)
                batched_kwargs = dict(c=[3,4,5], d=[6,7,8])
            Output:
                dict(a=[1,1,1], b=[2,2,2], c=[3,4,5], d=[6,7,8])
        """

        batched_solve_kwargs = batched_solve_kwargs or dict()
        for name, values in batched_solve_kwargs.items():
            assert len(values) == self.n_envs

        for name, value in common_solve_kwargs.items():
            assert name not in batched_solve_kwargs, \
                f'duplicated parameter {name}'
            batched_solve_kwargs[name] = [value] * self.n_envs

        if batched_solve_kwargs:
            # dictionary of lists -> list of dictionaries
            kwargs_names = batched_solve_kwargs.keys()
            solve_kwargs_per_agent = [
                dict(zip(kwargs_names, agent_kwargs_values))
                for agent_kwargs_values in zip(*batched_solve_kwargs.values())
            ]
        else:
            # No kwargs passed
            solve_kwargs_per_agent = [dict() for _ in range(self.n_envs)]
        return solve_kwargs_per_agent

    def _run_episode_batch(self, params, solve_kwargs_per_agent):  # pylint: disable=missing-param-doc
        """Runs a batch of episodes using the given network parameters.

        Args:
            params (Network-dependent): Network parameters.
            solve_kwargs_per_agent (list): List of kwargs passed to agents -
                i'th agent will be called with **solved_kwargs_per_agent[i]
                passed to solve().
        """
        raise NotImplementedError

    def run_episode_batch(
        self, params, batched_solve_kwargs=None, **common_solve_kwargs
    ):  # pylint: disable=missing-param-doc
        """Runs a batch of episodes using the given network parameters.

        Args:
            params (Network-dependent): Network parameters.
            batched_solve_kwargs (dict): Batched keyword arguments passed to
                Agent.solve(). This should be used if agents should get
                different parameters to solve(). If all agent are supposed to
                use the same kwargs, pass them via **common_solve_kwargs.
            **common_solve_kwargs (dict): Keyword arguments passed to
                Agent.solve().

        Returns:
            List of completed episodes (Agent/Trainer-dependent).
        """
        solve_kwargs_per_agent = self._prepare_solve_kwargs(
            batched_solve_kwargs, common_solve_kwargs)
        return self._run_episode_batch(params, solve_kwargs_per_agent)

    def close(self):
        """Cleans up the resources, e.g. shuts down the workers."""


class RequestHandler:
    """Handles requests from the agent coroutine to the network."""

    def __init__(self, network_fn):
        """Initializes RequestHandler.

        Args:
            network_fn (callable): Function () -> Network.
        """
        self.network_fn = network_fn

        self._network = None  # Lazy initialize if needed
        self._should_update_params = None

    def run_coroutine(self, episode_cor, params):  # pylint: disable=missing-param-doc
        """Runs an episode coroutine using the given network parameters.

        Args:
            episode_cor (coroutine): Agent.solve coroutine.
            params (Network-dependent): Network parameters.

        Returns:
            List of completed episodes (Agent/Trainer-dependent).
        """
        self._should_update_params = True

        try:
            request = next(episode_cor)
            while True:
                if isinstance(request, data.NetworkRequest):
                    request_handler = self._handle_network_request
                else:
                    request_handler = self._handle_prediction_request

                response = request_handler(request, params)
                request = episode_cor.send(response)
        except StopIteration as e:
            return e.value  # episodes

    def _handle_network_request(self, request, params):
        del request
        return self.network_fn, params

    def _handle_prediction_request(self, request, params):
        return self.get_network(params).predict(request)

    def get_network(self, params=None):
        if self._network is None:
            self._network = self.network_fn()
        if params is not None and self._should_update_params:
            self.network.params = params
            self._should_update_params = False
        return self._network
    network = property(get_network)
