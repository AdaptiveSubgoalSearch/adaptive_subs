import collections


Transition = collections.namedtuple('Trantision', [
    'state',
    'next_state',
    # Arbitrary info you want to pass from the goal builder to the caller.
    'info',
])


ShootingEpisode = collections.namedtuple('ShootingEpisode', [
    # List of transitions of the solving trajectory.
    'transitions',
    # Whether we solved the env or not.
    'done',
    # Number of random trajectories sampled.
    'n_trials',
    # Dict trajectory -> count.
    'trajectory_counts',
])


def sample_trajectory(init_state, goal_builder, time_limit):
    cur_state = init_state
    steps_taken = 0
    done = False
    transitions = []
    while not done and steps_taken < time_limit:
        goal = goal_builder.build_goal(cur_state)
        if goal is None:
            break
        next_state, done, info = goal
        transitions.append(Transition(
            state=cur_state, next_state=next_state, info=info
        ))

        cur_state = next_state
        steps_taken += 1

    return transitions, done


def random_shooting_solve(init_state, goal_builder, time_limit, n_trajectories, state_to_str=None):
    """Solves problem using random shooting.

    Args:
        init_state: Initial state.
        goal_builder: Goal builder implementing GoalBuilderForShooting interface.
        time_limit (int): Maximum number of subgoal-level steps to perform per trajectory.
        n_trajectories (int): Maximum number of random trajectories to sample.
        state_to_str: Optional function state->str used to determine if 2 trajectories
            are equal or not. Used to fill ShootingEpisode.trajectory_counts field.

    Returns:
        ShootingEpisode object (see the definition of this namedtuple).
    """

    trajectory_counts = collections.defaultdict(lambda: 0)
    for i in range(n_trajectories):
        transitions, done = sample_trajectory(init_state, goal_builder, time_limit)
        if state_to_str:
            trajectory_repr = tuple(
                state_to_str(transition.next_state)
                for transition in transitions
            )
            trajectory_counts[trajectory_repr] += 1
        if done:
            return ShootingEpisode(
                transitions=transitions,
                done=True,
                n_trials=i + 1,
                trajectory_counts=trajectory_counts,
            )
    return ShootingEpisode(
        transitions=None,
        done=False,
        n_trials=n_trajectories,
        trajectory_counts=trajectory_counts,
    )
