from pathlib import Path
from typing import Callable, Tuple, Dict, Optional
import tensorflow as tf
from tensorflow import Tensor


class DataPipeFactory:
    def __init__(self, tfrecord_path, ref_audio_path, word_information_path, cache=None):
        self.tfrecord_path: Path = Path(tfrecord_path)
        self.ref_audio_path: Path = Path(ref_audio_path)
        self.word_information_path: Path = Path(word_information_path)
        self.__cache_status = False
        if not self.tfrecord_path.exists():
            raise FileNotFoundError(f"tfrecord_path {tfrecord_path} not found")
        if not self.ref_audio_path.exists():
            raise FileNotFoundError(f"ref_audio_path {ref_audio_path} not found")
        if not self.word_information_path.exists():
            raise FileNotFoundError(f"word_information_path {word_information_path} not found")
        self.__cache = str(cache)
        self.__pairs: tf.int32 = 1
        self.__available_voice = 4
        self.__mel_bins = 80
        self.__raw_data: tf.data.Dataset = self.__generate_raw_data()

    # create the parser function to parse the serialized generated above
    @staticmethod
    def parse_function(serialized_example: tf.string) -> Dict:
        # Define a dict with the data-names and types we expect to find in the
        # serialized example.
        features = {
            'RecordName': tf.io.FixedLenFeature([], tf.string),
            'AudioSegment': tf.io.FixedLenFeature([], tf.string),
            'SampleRate': tf.io.FixedLenFeature([], tf.int64),
            'Sentence': tf.io.FixedLenFeature([], tf.string),
            'WordStart': tf.io.FixedLenFeature([], tf.string),
            'WordDuration': tf.io.FixedLenFeature([], tf.string),
            'MatchSegment': tf.io.FixedLenFeature([], tf.string),
            'MatchReference': tf.io.FixedLenFeature([], tf.string),
        }
        # Parse the input tf.Example proto using the dictionary above.
        e = tf.io.parse_single_example(serialized_example, features)
        # Convert the serialized tensor to tensor
        e['AudioSegment'] = tf.io.parse_tensor(e['AudioSegment'], out_type=tf.int16)
        e['Sentence'] = tf.io.parse_tensor(e['Sentence'], out_type=tf.int64)
        e['WordStart'] = tf.io.parse_tensor(e['WordStart'], out_type=tf.float32)
        e['WordDuration'] = tf.io.parse_tensor(e['WordDuration'], out_type=tf.float32)
        e['MatchSegment'] = tf.io.parse_tensor(e['MatchSegment'], out_type=tf.int64)
        e['MatchReference'] = tf.io.parse_tensor(e['MatchReference'], out_type=tf.int64)
        passage_id = tf.strings.split(e['RecordName'], sep='_')[3]
        # convert tf.string to int
        passage_id = tf.strings.to_number(passage_id, out_type=tf.int32) % 100000
        # convert to tf.string
        e['passage_id'] = tf.strings.as_string(passage_id)
        return e

    def __first_map_builder(self) -> Callable:
        get_mfcc = self.get_mfcc
        ref_audio_path = str(self.ref_audio_path.absolute())
        word_information_path = str(self.word_information_path.absolute())
        available_voice = self.__available_voice

        def created_map(e: Dict) -> Dict:
            a = {'stu_mfcc': get_mfcc(e['AudioSegment'], e['SampleRate'])}
            ref_audio = tf.io.parse_tensor(tf.io.read_file(ref_audio_path + '/' + e['passage_id'] + '.tfs'),
                                           out_type=tf.int16)
            a['ref_mfcc'] = get_mfcc(ref_audio, e['SampleRate'])
            passage_word = tf.io.parse_tensor(
                tf.io.read_file(word_information_path + '/' + e['passage_id'] + '_word.tfs'), out_type=tf.int64)
            reference_time = tf.io.parse_tensor(
                tf.io.read_file(word_information_path + '/' + e['passage_id'] + '_ref.tfs'), out_type=tf.float32)
            a['valid_stu_start'] = tf.gather(e['WordStart'], e['MatchSegment'])
            a['valid_stu_duration'] = tf.gather(e['WordDuration'], e['MatchSegment'])

            a['valid_ref_word'] = tf.gather(passage_word, e['MatchReference'], batch_dims=1)
            a['valid_ref_start'] = tf.gather(reference_time[..., 0], e['MatchReference'], batch_dims=1)
            a['valid_ref_duration'] = tf.gather(reference_time[..., 1], e['MatchReference'], batch_dims=1)

            a['RecordName'] = e['RecordName']
            a['passage_id'] = e['passage_id']
            a['MatchSegment'] = e['MatchSegment']
            a['MatchReference'] = e['MatchReference']

            a['stu_mfcc'].set_shape([None, 80])
            a['ref_mfcc'].set_shape([available_voice, None, 80])
            a['valid_stu_start'].set_shape([available_voice, None])
            a['valid_stu_duration'].set_shape([available_voice, None])
            a['valid_ref_word'].set_shape([available_voice, None])
            a['valid_ref_start'].set_shape([available_voice, None])
            a['valid_ref_duration'].set_shape([available_voice, None])
            a['MatchSegment'].set_shape([available_voice, None])
            a['MatchReference'].set_shape([available_voice, None])
            return a

        return created_map

    def __generate_raw_data(self) -> tf.data.Dataset:
        self.__raw_data = tf.data.TFRecordDataset(self.tfrecord_path, compression_type='GZIP') \
            .map(self.parse_function, num_parallel_calls=tf.data.AUTOTUNE) \
            .map(self.__first_map_builder(), num_parallel_calls=tf.data.AUTOTUNE) \
            .prefetch(tf.data.AUTOTUNE)
        return self.__raw_data

    @staticmethod
    @tf.function
    def get_mfcc(pcm: int,
                 sample_rate: int = 16000,
                 frame_length: int = 1024) -> tf.float32:
        # Implement the mel-frequency coefficients (MFC) from a raw audio signal.
        pcm = tf.cast(pcm, tf.float32) / tf.int16.max
        st_fft = tf.signal.stft(pcm, frame_length=frame_length, frame_step=frame_length // 8, fft_length=frame_length)
        spectrograms = tf.abs(st_fft)
        # Warp the linear scale spectrograms into the mel-scale.
        num_spectrogram_bins = frame_length // 2 + 1
        lower_edge_hertz, upper_edge_hertz, num_mel_bins = 80.0, 7600.0, 80
        linear_to_mel_weight_matrix = \
            tf.signal.linear_to_mel_weight_matrix(num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz,
                                                  upper_edge_hertz)
        mel_spectrograms = tf.einsum('...t,tb->...b', spectrograms, linear_to_mel_weight_matrix)
        log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)
        return log_mel_spectrograms

    @staticmethod
    def __pair_mapping(main: dict, counter: dict) -> dict:
        sample_dict = {}
        random_ref_voice_id = tf.random.uniform(shape=[], minval=0, maxval=tf.shape(counter['ref_mfcc'])[0],
                                                dtype=tf.int32)
        counter_random_ref_voice_id = tf.random.uniform(shape=[], minval=0, maxval=tf.shape(counter['ref_mfcc'])[0],
                                                        dtype=tf.int32)
        sample_dict['stu_mfcc'] = main['stu_mfcc']
        sample_dict['ref_mfcc'] = main['ref_mfcc'][random_ref_voice_id]
        sample_dict['valid_stu_start'] = main['valid_stu_start'][random_ref_voice_id]
        sample_dict['valid_stu_duration'] = main['valid_stu_duration'][random_ref_voice_id]
        sample_dict['valid_ref_word'] = main['valid_ref_word'][random_ref_voice_id]
        sample_dict['valid_ref_start'] = main['valid_ref_start'][random_ref_voice_id]
        sample_dict['valid_ref_duration'] = main['valid_ref_duration'][random_ref_voice_id]

        sample_dict['counter_ref_mfcc'] = counter['ref_mfcc'][counter_random_ref_voice_id]

        # Sample same mount of period from counter that match the main
        # Get the range of word under main
        main_word_range = tf.shape(sample_dict['valid_ref_word'])
        # Sample same mount of period from counter that match the main Generate same amount of random integer match
        # up the range of main_word_range counter_word_index = tf.random.uniform(shape=main_word_range, minval=0,
        # maxval=tf.shape(counter['valid_ref_word'][counter_random_ref_voice_id])[0], dtype=tf.int32)
        shuffled_index = tf.random.shuffle(
            tf.range(tf.shape(counter['valid_ref_word'][counter_random_ref_voice_id])[0]))
        if tf.shape(shuffled_index)[0] > main_word_range[0]:
            counter_word_index = shuffled_index[:main_word_range[0]]
        else:
            counter_word_index = tf.random.uniform(shape=main_word_range, minval=0, maxval=
            tf.shape(counter['valid_ref_word'][counter_random_ref_voice_id])[0], dtype=tf.int32)
            # replace the value in the range of shuffled_index with the value in counter_word_index
            counter_word_index = \
                tf.tensor_scatter_nd_update(
                    counter_word_index,
                    tf.range(tf.shape(shuffled_index)[0])[..., tf.newaxis],
                    shuffled_index)
        # Sample data using counter_word_index
        sample_dict['counter_valid_ref_word'] = \
            tf.gather(counter['valid_ref_word'][counter_random_ref_voice_id], counter_word_index)
        sample_dict['counter_valid_ref_start'] = \
            tf.gather(counter['valid_ref_start'][counter_random_ref_voice_id], counter_word_index)
        sample_dict['counter_valid_ref_duration'] = \
            tf.gather(counter['valid_ref_duration'][counter_random_ref_voice_id], counter_word_index)
        # determine if counter_valid_ref_word with main_valid_ref_word match up if match up return 1. else return -1.
        sample_dict['counter_word_match'] = tf.where(tf.equal(sample_dict['counter_valid_ref_word'],
                                                              sample_dict['valid_ref_word']), 1., -1.)
        sample_dict['counter_pool_index'] = counter_word_index
        return sample_dict

    @staticmethod
    def __single_mapping(main: dict) -> dict:
        sample_dict = {}
        random_ref_voice_id = tf.random.uniform(shape=[], minval=0, maxval=tf.shape(main['ref_mfcc'])[0],
                                                dtype=tf.int32)
        sample_dict['stu_mfcc'] = main['stu_mfcc']
        sample_dict['ref_mfcc'] = main['ref_mfcc'][random_ref_voice_id]
        sample_dict['valid_stu_start'] = main['valid_stu_start'][random_ref_voice_id]
        sample_dict['valid_stu_duration'] = main['valid_stu_duration'][random_ref_voice_id]
        sample_dict['valid_ref_word'] = main['valid_ref_word'][random_ref_voice_id]
        sample_dict['valid_ref_start'] = main['valid_ref_start'][random_ref_voice_id]
        sample_dict['valid_ref_duration'] = main['valid_ref_duration'][random_ref_voice_id]

        # Sample same mount of period from counter that match the main
        # Get the range of word under main
        main_word_range = tf.shape(sample_dict['valid_ref_word'])
        # Sample same mount of period from counter that match the main Generate same amount of random integer match
        # up the range of main_word_range counter_word_index = tf.random.uniform(shape=main_word_range, minval=0,
        # maxval=tf.shape(counter['valid_ref_word'][counter_random_ref_voice_id])[0], dtype=tf.int32)
        return sample_dict

    def pre_save(self) -> None:
        self.__raw_data.save(self.__cache, compression='GZIP')
        self.__cache_status = True
        self.__raw_data = tf.data.Dataset.load(self.__cache).load(self.__cache)
        print(f'Cache saved to {self.__cache}')

    # if cache folder is not exist create it
    # or load the cache
    # if load fail, create a new cache
    def try_save(self) -> None:
        if not Path(self.__cache).exists():
            self.pre_save()
        else:
            try:
                self.__raw_data = tf.data.Dataset.load(self.__cache)
                self.__cache_status = True
                print(f'Load cache from {self.__cache}')
            except Exception as e:
                print(f'Load cache failed, create new cache')
                print(e)
                self.pre_save()

    # if cache loaded do not load again
    # if cache not loaded, load it
    def get_raw_data(self) -> tf.data.Dataset:
        if not self.__cache_status and Path(self.__cache).exists():
            print(f'Load cache from {self.__cache}')
            self.__raw_data = tf.data.Dataset.load(self.__cache, compression='GZIP')
            self.__cache_status = True
        return self.__raw_data

    def get_pair_data(self) -> tf.data.Dataset:
        return self.get_raw_data().apply(self.__pair_map_handle(self.__pairs))

    def __batching_handle(self, batch_size: int) -> Callable[[tf.data.Dataset], tf.data.Dataset]:
        def handle(ds):
            return ds \
                .padded_batch(batch_size,
                              padding_values={k: tf.cast(-1, v.dtype) if v.dtype != tf.string else '' for k, v in
                                              ds.element_spec.items()}, drop_remainder=True) \
                .prefetch(tf.data.experimental.AUTOTUNE)

        return handle

    def __pair_map_handle(self, pairs: int,
                          deterministic: bool = True) \
            -> Callable[[tf.data.Dataset], tf.data.Dataset]:
        if pairs == 1:
            def handle(ds):
                return ds.map(self.__single_mapping, num_parallel_calls=tf.data.AUTOTUNE, deterministic=deterministic) \
                    .shuffle(buffer_size=10, reshuffle_each_iteration=True)
            pass
        else:
            def handle(ds):
                tuple_of_pairs = tuple(ds.shuffle(20, reshuffle_each_iteration=True) for _ in range(pairs))
                comb_data = tf.data.Dataset.zip(tuple_of_pairs).filter(lambda x, y: x["RecordName"] != y["RecordName"])
                return comb_data.map(self.__pair_mapping, num_parallel_calls=tf.data.AUTOTUNE,
                                     deterministic=deterministic) \
                    .shuffle(buffer_size=10, reshuffle_each_iteration=True)

        return handle

    def k_fold(self, total_fold: int,
               fold_index: int,
               batch_size: int,
               addition_map: Optional = None,
               deterministic: bool = False) \
            -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        if fold_index >= total_fold:
            raise ValueError("fold_index must be less than total_fold")
        indexed_data = self.get_raw_data().enumerate()
        train_data = indexed_data \
            .filter(lambda index, _: index % total_fold != fold_index) \
            .map(lambda _, data: data, num_parallel_calls=tf.data.AUTOTUNE, deterministic=deterministic) \
            .apply(self.__pair_map_handle(self.__pairs, deterministic=deterministic)) \
            .apply(self.__batching_handle(batch_size))

        test_data = indexed_data \
            .filter(lambda index, _: index % total_fold == fold_index) \
            .map(lambda _, data: data, num_parallel_calls=tf.data.AUTOTUNE, deterministic=deterministic) \
            .apply(self.__pair_map_handle(self.__pairs, deterministic=deterministic)) \
            .apply(self.__batching_handle(batch_size))
        if addition_map is not None:
            train_data = train_data.map(addition_map, num_parallel_calls=tf.data.AUTOTUNE, deterministic=deterministic).prefetch(tf.data.AUTOTUNE)
            test_data = test_data.map(addition_map, num_parallel_calls=tf.data.AUTOTUNE, deterministic=deterministic).prefetch(tf.data.AUTOTUNE)
        return train_data, test_data

    def get_batch_data(self,
                       batch_size: int,
                       addition_map: Optional = None,
                       deterministic=False) -> tf.data.Dataset:
        if addition_map is not None:
            return self.get_raw_data().apply(self.__pair_map_handle(self.__pairs, deterministic=deterministic)).apply(
                self.__batching_handle(batch_size)) \
                .map(addition_map, num_parallel_calls=tf.data.AUTOTUNE, deterministic=deterministic).prefetch(tf.data.AUTOTUNE)
        else:
            return self.get_raw_data().apply(self.__pair_map_handle(self.__pairs, deterministic=deterministic)).apply(
                self.__batching_handle(batch_size))