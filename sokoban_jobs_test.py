from internal_run import internal_run

def test_solve():
    solved_metrics, finished_time = internal_run('experiments/solve.py')
    assert float(finished_time) < 60., 'Solving takes too long'
    assert solved_metrics.get_value('rate') == 1.0, 'Did not solve some boards'

def test_train():
    finished_time = internal_run('experiments/train.py')
    assert float(finished_time) < 60., 'Training takes too long'

def test_draw_tree():
    _, finished_time = internal_run('experiments/draw_tree.py')
    assert float(finished_time) < 25., 'Drawing takes too long'

def test_create_samples():
    finished_time = internal_run('experiments/create_samples.py')
    print(f'FT = {finished_time}')
    assert float(finished_time) < 25., 'Creating samples takes too long'