import tensorflow as tf
import numpy as np
from pathlib import Path
from argparse import ArgumentParser

def read_save_handle(read_path: Path, write_path: Path) -> str:
    """
    Read the data from read_path and save it to write_path
    :param read_path: a PathLib path to the file to be read which has extension .tfs
    :param write_path: a PathLib path to the file to be written which has extension .npy
    :return: name of the file
    """
    # Check if read_path is extension .tfs
    if read_path.suffix != '.tfs':
        raise ValueError('read_path must have extension .tfs')
    # Check if write_path is extension .npy
    if write_path.suffix != '.npy':
        raise ValueError('write_path must have extension .npy')
    # Read the data
    tfs = tf.io.read_file(str(read_path))
    # Parse the data
    data = tf.io.parse_tensor(tfs, tf.float32).numpy()
    # Save the data using numpy
    np.save(str(write_path), data)
    # Return the name of the file without extension
    return write_path.stem
    pass


def write_path_generator(write_prefix: Path, read_file: Path) -> Path:
    """
    Generate the write path by appending the read_file name (without an extension) to write_prefix
    :param write_prefix: an PathLib path to the directory to be written
    :param read_file: a PathLib path to the file to be read
    :return: write_path: a PathLib path to the file to be written
    """
    # Check if write_prefix is a directory
    if not write_prefix.is_dir():
        raise ValueError('write_prefix must be a directory')
    # Check if read_file is a file
    if not read_file.is_file():
        raise ValueError('read_file must be a file')
    # Check if read_file has extension .tfs
    if read_file.suffix != '.tfs':
        raise ValueError('read_file must have extension .tfs')
    # Generate the write_path
    write_path = write_prefix / (read_file.stem + '.npy')
    # Return the write_path
    return write_path


if __name__ == "__main__":
    # Parse the arguments
    parser = ArgumentParser()
    parser.add_argument('--read_dir', type=str, required=True, help='a directory to read the files')
    parser.add_argument('--write_dir', type=str, required=True, help='a directory to write the files')
    args = parser.parse_args()
    # Convert the read_dir and write_dir to PathLib paths
    read_dir = Path(args.read_dir)
    write_dir = Path(args.write_dir)
    # Check if read_dir is a directory
    if not read_dir.is_dir():
        raise ValueError('read_dir must be a directory')
    # Check if write_dir is a directory
    if not write_dir.is_dir():
        raise ValueError('write_dir must be a directory')

    # Print task description with details
    print(f'Converting .tfs files from dir {read_dir} to .npy in dir{write_dir}')
    # Get the list of files in read_dir
    files = list(read_dir.glob('*.tfs'))
    # Iterate through the files
    for file in files:
        # Generate the write_path
        write_path = write_path_generator(write_dir, file)
        # Read the data from read_path and save it to write_path
        read_save_handle(file, write_path)
        # Print the file name
        print(f'File {file.stem} converted')
    print('All files converted')
