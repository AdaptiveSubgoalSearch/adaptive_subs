import argparse
import logging
import os
import random
import sys
from typing import List

import joblib
import numpy as np
from utils.utils_sokoban import many_states_to_fig


# Utils


def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def get_all_chunk_files(root: str, prefix: str):
    all_files = []
    for folder, _, files in os.walk(root):
        all_files += [os.path.join(folder, f)
                      for f in files if f.startswith(prefix)]
    random.shuffle(all_files)
    return all_files


# Valid


def is_leak(train_part, entries_from_valid):
    for valid_entry in entries_from_valid:
        count = sum([1 if np.array_equal(valid_entry['start'], other['start']) and np.array_equal(
            valid_entry['subgoal'], other['subgoal']) else 0 for other in train_part])
        if count > 0:
            return True
    return False


def check_no_leak(train_parts, valid_parts, sanity_check_number, threads):
    valid_chunk = random.choice(valid_parts)

    entries_from_valid = random.sample(
        valid_chunk, k=min(sanity_check_number, len(valid_chunk)))

    results = joblib.Parallel(n_jobs=threads, verbose=10000)(
        joblib.delayed(is_leak)(train_part, entries_from_valid)
        for train_part in train_parts
    )
    # Flatten
    result = list(map(lambda flag: 1 if flag else 0, results))
    count = sum(result)
    if count > 0:
        raise Exception(
            f'Found {count} occurences (from {len(result)}) of given valid sample in train dataset. Aborting')


def sanity_check(parts, unique_check: int):
    ds = random.choice(parts)
    check = min(unique_check, len(ds))
    entries = random.sample(ds, check)
    set_entry = set(hash_entry(entry) for entry in entries)
    if len(set_entry) != len(entries):
        raise Exception(
            f'Found {len(entries) - len(set_entry)} occurences of given sample in dataset.')


# Dataset Gather


def make_uniform(ds):
    """Make dataset uniform with regards to trajectories idxs
    """
    uniform_ds = []

    # Mapping from board_id -> dataset entries
    unique_board_ids = set([entry['root_board_id'] for entry in ds])

    board2entries = {
        boardid: [] for boardid in unique_board_ids
    }

    for entry in ds:
        board2entries[entry['root_board_id']].append(entry)

    # Sample uniformly
    median_wrt_board_id = int(
        np.median(np.array([len(v) for v in board2entries.values()])))

    # Update median in order to get more samples
    median_wrt_board_id = median_wrt_board_id

    for entries_per_board in board2entries.values():
        # We take smaller value
        size_of_current_sample = min(
            len(entries_per_board), median_wrt_board_id)
        sample = random.choices(entries_per_board, k=size_of_current_sample)
        uniform_ds.append(sample)

    # Flatten
    uniform_ds = [entry for d in uniform_ds for entry in d]

    # Return datasets with stats stats
    return uniform_ds, len(board2entries), median_wrt_board_id


def gather_part(chunk_file_paths: str):
    ds_non_uniform = []
    for chunk_file_path in chunk_file_paths:
        try:
            memory = joblib.load(chunk_file_path)
        except:
            print(f'Omitting {chunk_file_path} due to fail during loading')
        for ver_memory in memory.verificator_memory:
            ds_non_uniform += ver_memory.data

    ds, boards, median = make_uniform(ds_non_uniform)

    p0 = [entry for entry in ds if entry['probability'] == 0.0]
    p1 = [entry for entry in ds if entry['probability'] == 1.0]

    print(f'Distribution before balancing... p == 0.0: {len(p0)} ({round(len(p0)/len(ds) * 100, 2)}%), p == 1.0: {len(p1)} ({round(len(p1)/len(ds) * 100, 2)})%')
    print('Balancing')

    if len(p1) > len(p0):
        p1 = random.sample(p1, len(p0))
    elif len(p0) > len(p1):
        p0 = random.sample(p0, len(p1))

    print(f'Distribution before balancing... p == 0.0: {len(p0)} ({round(len(p0)/len(ds) * 100, 2)}%), p == 1.0: {len(p1)} ({round(len(p1)/len(ds) * 100, 2)})%')
    ds = p0 + p1
    random.shuffle(ds)
    print(f'Size of the dataset after balancing: {len(ds)}')

    # Drop stats
    return len(p1), len(p0), boards, median, ds


def gather_dataset(chunk_file_paths: List[str], threads: int, expected_number_of_samples: int):
    batch_size = 500
    epochs = 5
    print(f'Epochs: {epochs}')
    # if current number of samples > upper_bound_factor*expected_number_of_samples, stop gathering
    upper_bound_factor = 2
    chunk_file_path_epochs = split(chunk_file_paths, epochs)
    break_condition = upper_bound_factor*expected_number_of_samples

    ds = []
    for chunk_file_path_epoch in chunk_file_path_epochs:
        print(f'Start processing epoch with {len(chunk_file_path_epoch)}')
        chunk_file_path_packs = chunks(chunk_file_path_epoch, batch_size)
        try:
            results = joblib.Parallel(n_jobs=threads, verbose=10000)(
                joblib.delayed(gather_part)(chfs)
                for chfs in chunk_file_path_packs
            )
        except KeyboardInterrupt:
            exit(0)
        except:
            print('Skip due to struct.error')
            continue
        # Show stats
        reachables, nonreachables = 0, 0
        medians = []
        boards_total = 0

        for (r, nr, bt, m, d) in results:
            reachables += r
            nonreachables += nr
            boards_total += bt
            medians.append(m)

            # Update dataset
            ds += d
        print(f'Current number of samples: {len(ds)}. Required: {break_condition} as least')
        if len(ds) > break_condition:
            print('Already gathered enough samples', len(ds))
            break

    print(f'Gathered {reachables+nonreachables} samples in total from {boards_total} boards.')
    print(f'Reachable: {reachables} ({round(100*reachables/(reachables+nonreachables),2)}%)')
    print(f'Nonreachable: {nonreachables} ({round(100*nonreachables/(reachables+nonreachables),2)}%)')
    print(f'Medians:', medians)

    return ds


# Clean Dataset


# already defined above
# def split(a, n):
#     k, m = divmod(len(a), n)
#     return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))


def hash_entry(entry):
    return np.concatenate([entry['start'], entry['subgoal']], axis=-1).tostring()


def remove_repeats(ds, label):
    ds_without_repeats = list(
        {hash_entry(entry): entry for entry in ds}.values())
    diff = len(ds) - len(ds_without_repeats)
    if diff > 0:
        print(
            f'Found {diff} repeats in {label} dataset ({round(100*diff/len(ds), 2)}%). Removed them')
    else:
        print(f'Repeats not found in {label} dataset.')
    random.shuffle(ds_without_repeats)
    return ds_without_repeats


def remove_leaks(ds_train, ds_valid):
    train_hashed = {hash_entry(entry) for entry in ds_train}

    ds_valid_filtered = list(
        filter(lambda entry: hash_entry(entry) not in train_hashed, ds_valid))
    diff = len(ds_valid) - len(ds_valid_filtered)
    print('Difference is', diff)
    if diff > 0:
        print(f'Found {diff} leaks. Removed them')
    else:
        print('Leaks not found')

    return ds_valid_filtered


def clean(ds_train, ds_valid, splits_number):
    print(f'Before cleaning: valid {len(ds_valid)} train {len(ds_train)}')

    # Repeats
    ds_train = remove_repeats(ds_train, 'train')
    ds_valid = remove_repeats(ds_valid, 'valid')

    # Leaks
    ds_valid = remove_leaks(ds_train, ds_valid)

    # Summary
    print(f'After cleaning: valid {len(ds_valid)} train {len(ds_train)}')

    # Split
    valid_parts = list(split(ds_valid, splits_number))
    train_parts = list(split(ds_train, splits_number))
    return train_parts, valid_parts


# Draw


def draw_from_ds(ds, n: int, label: str):
    p0 = [entry for entry in ds if entry['probability'] == 0.0]
    p1 = [entry for entry in ds if entry['probability'] == 1.0]

    print('len(entries where p = 0)', len(p0))
    print('len(entries where p = 1)', len(p1))

    p0sampled = random.sample(p0, min(n, len(p0)))
    p1sampled = random.sample(p1, min(n, len(p1)))

    for k, entry in enumerate(p0sampled+p1sampled):
        fig = many_states_to_fig(
            [entry['start'], entry['subgoal']], ['start', 'subgoal'])
        fig.savefig(
            f'{label}_reachable_img_idx{k}_prob{int(entry["probability"])}')


# Main pipeline


def dataset_pipeline(jobs_root: str, valid_percent: float, prefix: str, unique_check: int, split_number: int,
                     threads: int, output: str, draw_number: int, target_size: int):
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    # Collect chunk paths
    logging.info('Collect chunk paths')
    chunk_files = get_all_chunk_files(jobs_root, prefix)
    print(f'Found {len(chunk_files)} chunk files')
    random.shuffle(chunk_files)
    train_samples = len(chunk_files) - int(valid_percent*len(chunk_files))

    chunk_files_train = chunk_files[:train_samples]
    chunk_files_valid = chunk_files[train_samples:]

    # Gather dataset chunks
    target_valid_size = int(target_size*valid_percent)
    target_train_size = target_size - target_valid_size

    logging.info('Gather dataset chunks')
    ds_train = gather_dataset(chunk_files_train, threads, target_train_size)
    ds_valid = gather_dataset(chunk_files_valid, threads, target_valid_size)

    # Cut dataset to target size

    if len(ds_train) > target_train_size:
        print(
            f'Shortening train size from {len(ds_train)} to {target_train_size}')
        random.shuffle(ds_train)
        ds_train = ds_train[:target_train_size]

    if len(ds_valid) > target_valid_size:
        print(
            f'Shortening train size from {len(ds_valid)} to {target_valid_size}')
        random.shuffle(ds_valid)
        ds_valid = ds_valid[:target_valid_size]

    # Clean dataset
    logging.info('Clean dataset')
    ds_train_parts, ds_valid_parts = clean(ds_train, ds_valid, split_number)

    # Valid
    logging.info('Validating if no leak')
    check_no_leak(ds_train_parts, ds_valid_parts, unique_check, threads)

    logging.info('Validating if no repeat')
    sanity_check(ds_train_parts+ds_valid_parts, unique_check)

    # Visualize
    logging.info('Draw reachable/unreachable plots')

    output_train = f'{output}_train'
    output_valid = f'{output}_valid'
    draw_from_ds(ds_train_parts[0], draw_number, output_train)
    draw_from_ds(ds_valid_parts[0], draw_number, output_valid)

    # Dump
    for idx, (train_part, valid_part) in enumerate(zip(ds_train_parts, ds_valid_parts)):
        joblib.dump(train_part, f'{output_train}_{idx}', compress=4)
        joblib.dump(valid_part, f'{output_valid}_{idx}', compress=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--jobs-root', help='Path solver jobs', required=True, type=str)
    parser.add_argument(
        '--valid-percent', help='Path solver jobs', required=False, default=0.1, type=float)
    parser.add_argument(
        '--prefix', help='Prefix of dataset chunks', required=True, type=str)
    parser.add_argument(
        '--unique-check', help='Number of files to check if they are unique', required=False, default=100, type=int)
    parser.add_argument(
        '--split-number', help='Number of parts', required=False, default=10, type=int)
    parser.add_argument(
        '--threads', help='Threads', required=False, default=8, type=int)
    parser.add_argument(
        '--draw-number', help='Draw number', required=False, default=20, type=int)
    parser.add_argument(
        '--target-size', help='Number of total samples in dataset', required=True, type=int)
    parser.add_argument(
        '--output', help='Prefix name for dataset and images', required=True, type=str)
    args = parser.parse_args()

    dataset_pipeline(args.jobs_root, args.valid_percent, args.prefix,
                     args.unique_check, args.split_number, args.threads, args.output, args.draw_number, args.target_size)
