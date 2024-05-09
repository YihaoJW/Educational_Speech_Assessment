import tensorflow as tf
from pathlib import Path


class ProsodyDataPipeFactory:
    """
    A class to handle the data pipeline for prosody prediction and return tf.data.Dataset objects
    All feature is pre generated and stored as tfs (tensorflow serialized tensors) files
    We need to read two folders, one contains the student features, the other contains the siri reference features
    Label is stored in a CSV file we need to pre-process it when Factory is initialized
    All directories should is using pathlib.Path object to represent
    """

    def __init__(self, student_folder: Path,
                 siri_folder: Path,
                 score_record: Path,
                 batch_size: int,
                 eval_mode: bool = False):
        """
        Initialize the factory and verify the folder and file are all exist folder are all Path object
        """
        self.student_folder = student_folder.resolve()
        assert self.student_folder.is_dir(), "Student folder is not a directory or not exist"
        self.siri_folder = siri_folder.resolve()
        assert self.siri_folder.is_dir(), "Siri folder is not a directory or not exist"
        self.score_record = score_record.resolve()
        assert self.score_record.is_file(), "score file is not a file or not exist"
        self.batch_size = batch_size
        self.__parse_example = {
            'filename': tf.io.FixedLenFeature([], tf.string),
            'rating': tf.io.FixedLenFeature([], tf.int64),
            'rating_max': tf.io.FixedLenFeature([], tf.float32),
            'rating_min': tf.io.FixedLenFeature([], tf.float32),
            'passage_id': tf.io.FixedLenFeature([], tf.int64),
            'student_id': tf.io.FixedLenFeature([], tf.int64)
        }
        self.__dataset = tf.data.TFRecordDataset([self.score_record])
        self.eval_mode = eval_mode
        self.__raw_data = None

    def __parse_function(self):
        # Parse the input tf.Example proto using the dictionary above.
        exp = self.__parse_example

        def __parse_example(example_proto):
            return tf.io.parse_single_example(example_proto, exp)

        return __parse_example

    def get_raw(self):
        """
        Iterate through the tfrecord files and return a tf.data.Dataset object
        """
        self.raw_data = self.__dataset.map(self.__parse_function())
        return self.raw_data

    def get_main_dataset(self):
        """
        Return the main tf.data.Dataset object
        """
        if self.eval_mode:
            return (self.get_raw().map(self.map_function_generator(), num_parallel_calls=tf.data.AUTOTUNE)
                    .padded_batch(self.batch_size, padding_values=((-1.0, (-1.0, -1.0, -1.0, -1.0)), -1.0),
                                  padded_shapes=(([None, 128],
                                                  ([None, 128], [None, 128], [None, 128], [None, 128])), [3]), drop_remainder=False)
                    .prefetch(tf.data.AUTOTUNE))
        else:
            return (self.get_raw().shuffle(5000)
                    .map(self.map_function_generator(), num_parallel_calls=tf.data.AUTOTUNE)
                    .padded_batch(self.batch_size, padding_values=((-1.0, (-1.0, -1.0, -1.0, -1.0)), -1.0),
                                  padded_shapes=(([None, 128],
                                                  ([None, 128], [None, 128], [None, 128], [None, 128])), [3]), drop_remainder=True)
                    .prefetch(tf.data.AUTOTUNE))

    def map_function_generator(self):
        """
        A function to map the tf.data.Dataset object to the final form
        """
        student_dir = str(self.student_folder)
        siri_dir = str(self.siri_folder)

        def rp(tensor):
            if tensor.shape.ndims is None or tensor.shape[1] is None:
                tensor.set_shape([None, 128])
            mask = tf.reduce_any(tensor != -1, axis=1)

            # Use this mask to filter out the rows
            filtered_tensor = tf.boolean_mask(tensor, mask)
            filtered_tensor.set_shape([None, 128])
            return filtered_tensor

        def mapping(data):
            filename = data['filename']
            passage_id = data['passage_id']
            rating = tf.cast(data['rating'], tf.float32)
            rating_max = data['rating_max']
            rating_min = data['rating_min']
            # covert passage_id to string and add extension ".tfs"
            passage_id_path = tf.strings.as_string(passage_id) + '.tfs'
            student_path = tf.strings.join([student_dir, filename], separator='/')
            siri_path = tf.strings.join([siri_dir, passage_id_path], separator='/')
            student = tf.io.parse_tensor(tf.io.read_file(student_path), out_type=tf.float32)
            # tensor student has padding with -1 remove it, we will add padding in padded_batch

            siri = tf.io.parse_tensor(tf.io.read_file(siri_path), out_type=tf.float32)
            # Mix rating rating_max and rating_min into a single tensor using tf.stack
            total_rating = tf.stack([rating, rating_min, rating_max])

            return (rp(student), (rp(siri[0]), rp(siri[1]), rp(siri[2]), rp(siri[3]))), total_rating

        return mapping
