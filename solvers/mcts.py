import gin

from alpacka.agents import StochasticMCTSAgent


@gin.configurable
def mcts_solve(goal_builder, value_function, init_state, time_limit, **kwargs):
    """
    Args:
        goal_builder: Goal builder complying to
            alpacka.agents.tree_search.GoalBuilder interface.
        value_function: Callable(list of states -> list of floats) computing
            a value for every given state.
        init_state: Initial state of environment to solve (env-dependent).
        time_limit: Maximal number of (subgoal-level) steps allowed.
        **kwargs: Kwargs to pass to StochasticMCTSAgent.

    Returns:
        Pair:
        * maybe_action_states: List of pairs (list of actions, environment state)
            describing subsequent states (or subgoals) and how to reach them.
            Initial state is not included. If mcts didn't find a solution,
            None is returned.
        * agent_info: MCTS-related metrics.
    """
    agent = StochasticMCTSAgent(
        value_function=value_function,
        goal_builder=goal_builder,
        **kwargs
    )
    episode = agent.solve(
        init_state=init_state, time_limit=time_limit
    )
    return episode
