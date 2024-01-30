from typing import Dict

import tensorflow as tf
from pathlib import Path

from tqdm import tqdm


class Test_DS_Root:
    """
    Base class for test dataset
    """

    def get_raw_ds(self):
        """
        get the raw dataset
        """
        raise NotImplementedError

    def get_ds(self):
        """
        get the dataset
        """
        raise NotImplementedError

    @staticmethod
    @tf.function(jit_compile=True)
    def get_mfcc(pcm: int,
                 sample_rate: int = 16000,
                 frame_length: int = 1024,
                 is_float=False) -> tf.float32:
        # Implement the mel-frequency coefficients (MFC) from a raw audio signal
        if not is_float:
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
    @tf.function
    def unpack(label):
        start = tf.RaggedTensor.from_tensor(label[:, :, 0], padding=-1.)
        duration = tf.RaggedTensor.from_tensor(label[:, :, 1], padding=-1.)
        return start, duration


class Test_DS_Factory_Siri(Test_DS_Root):
    """
    Factory class for creating test dataset
    Read the label which contains a tensorflow serialized matrix in tf.int64 format it has shape [num_of_speakers, num_of_frames]

    siri_voice_frame_path: path to the folder containing the original siri voice frames need to be mapped to mfc
    later the name has structure {passage_id}.tfs is has dtype it has shape [num_of_speakers, num_of_frames]
    tf.int16 siri_voice_label_path: path to the label folder
    file, it has structure {passage_id}_ref.tfs. it is a tf.float32 dtype matrix with shape [num_of_speakers,
    num_of_words,2], and it contains non-related tfs files in the folder that should be ignored
    the return dataset contains the mfc of the frame feature and the label of the speaker
    """

    def __init__(self, frame_feature_path, segment_feature_path, **kwargs):
        """
        :param frame_feature_path: it's a dir of frame feature of siri
        :param segment_feature_path: it's a dir of segment feature of siri
        """
        self.siri_voice_frame_path = Path(frame_feature_path)
        self.siri_voice_label_path = Path(segment_feature_path)

    def get_raw_ds(self):
        """
        get the raw dataset
        Read voice_frame and get the passage_id in its filename and use it to read the label file, return frame, label, and passage_id
        frame and label has the same batch shape which is num_of_speakers; we can assume it's will match up
        """
        passage_id_ds = tf.data.Dataset.list_files(str(self.siri_voice_frame_path / '*.tfs'))
        label_location_format = str(self.siri_voice_label_path / '{}_ref.tfs')

        def name_to_files_mapper(name):
            passage_id = tf.strings.split(tf.strings.split(name, '/')[-1], '.')[0]
            passage_id = tf.strings.to_number(passage_id, tf.int64)
            frame = tf.io.read_file(name)
            frame = tf.io.parse_tensor(frame, tf.int16)
            label = tf.io.read_file(tf.strings.format(label_location_format, passage_id))
            label = tf.io.parse_tensor(label, tf.float32)
            return frame, label[:, 1:, :], passage_id

        return passage_id_ds.map(name_to_files_mapper, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    @staticmethod
    def mask_correct(frame, passage_id, fix):
        value_f = frame[0]
        value_f = tf.where(tf.equal(value_f, -13.815510749816895), -1., value_f)
        return (value_f, frame[1]), passage_id, fix

    def get_final_ds(self):
        return self.get_raw_ds().map(lambda x, y, z:
                                     ((self.get_mfcc(x), self.unpack(y)), z, 'Siri'),
                                     num_parallel_calls=tf.data.experimental.AUTOTUNE).map(
            self.mask_correct, num_parallel_calls=tf.data.experimental.AUTOTUNE).prefetch(
            tf.data.experimental.AUTOTUNE)


class Test_DS_Factory_Student(Test_DS_Root):
    """
    Factory class for creating test dataset for students,
    First we load a tf dataset record from the file as base dataset.
    We need to read a label dictionary from using the RecordName information in the base dataset.
    The label is segment information of the audio file we need to load from the disk.
    Then we map the base dataset to the final dataset
    """

    def __init__(self, frame_feature_path, segment_feature_path, **kwargs):
        """
        :param frame_feature_path: a tensorflow dataset record file
        :param segment_feature_path: a dir to student segment information
        """
        self.student_dataset_record_path = Path(frame_feature_path)
        self.student_dataset_label_path = Path(segment_feature_path)
        self.seg_path = str(self.student_dataset_label_path)

    def parse_function(self, serialized_example: tf.string) -> Dict:
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
        ret = {'AudioSegment': tf.io.parse_tensor(e['AudioSegment'], out_type=tf.int16),
               'RecordName': tf.io.parse_tensor(e['RecordName'], tf.string)}
        # Convert the serialized tensor to tensor

        passage_id = tf.strings.split(e['RecordName'], sep='_')[3]
        # convert tf.string to int
        passage_id = tf.strings.to_number(passage_id, out_type=tf.int32) % 100000
        # convert to tf.string
        ret['passage_id'] = tf.strings.as_string(passage_id)
        ret['label'] = tf.io.parse_tensor(tf.io.read_file(self.seg_path + '/' + ret['RecordName'] + '.tfs'), tf.float32)

        return ret

    def get_raw_ds(self):
        """
        get the raw dataset
        Read voice_frame and get the passage_id in its filename and use it to read the label file, return frame, label, and passage_id
        frame and label has the same batch shape which is num_of_speakers; we can assume it's will match up
        """
        ds = tf.data.TFRecordDataset(str(self.student_dataset_record_path), compression_type='GZIP')
        return ds.map(self.parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(1)

    def get_final_ds(self):
        return self.get_raw_ds().map(lambda x:
                                     (
                                         (
                                             self.get_mfcc(x['AudioSegment']), self.unpack(x['label'])
                                         ),
                                         x['passage_id'], x['RecordName']
                                     ),
                                     num_parallel_calls=tf.data.experimental.AUTOTUNE).prefetch(
            tf.data.experimental.AUTOTUNE)


class Prosody_Data_Generation(Test_DS_Root):
    """
    Read a path of file that contain wav files,
    and it need to return mfcc of that wav file, mask of wav file, and name of the file
    """

    def __init__(self, audios_wav_path: str):
        self.audio_wav_path = Path(audios_wav_path)
        # secure check
        if not self.audio_wav_path.exists():
            raise FileNotFoundError(f"Audio wav path {self.audio_wav_path} is not existed")

    def get_raw_ds(self):
        """
        get the raw dataset
        Read voice_frame and get the passage_id in its filename and use it to read the label file, return frame, label, and passage_id
        frame and label has the same batch shape which is num_of_speakers; we can assume it's will match up
        """
        # get all the wav files in the folder using tensorflow dataset
        wav_files = tf.data.Dataset.list_files(str(self.audio_wav_path / '*.wav'))
        # return raw dataset with the wav file name
        return wav_files

    @staticmethod
    def parse_function(file_name: tf.string) -> Dict:
        """
        Read the wav file and return the mfcc and mask of that wav file
        """
        # read the wav file
        audio = tf.io.read_file(file_name)
        # decode the wav file
        audio, sample_rate = tf.audio.decode_wav(audio)
        # get the mfcc of the wav file on the single channel
        mfcc = Prosody_Data_Generation.get_mfcc(audio[:, 0], sample_rate, is_float=True)
        # get the mask of the single channel wav file is a 1d array with value 1.0
        mask = tf.ones_like(mfcc[:, 0])
        # get the file name
        file_name = tf.strings.split(tf.strings.split(file_name, '/')[-1], '.')[0]
        # return the mfcc, mask, and file name
        return {'mfcc': mfcc, 'mask': mask, 'file_name': file_name}

    def get_final_ds(self, batch_size=16):
        """
        get the final dataset with batch size batch_size, make sure mfcc and mask is padded to the same length with value -1 and 0
        """
        return self.get_raw_ds().map(self.parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE).padded_batch(
            batch_size, padded_shapes={'mfcc': [-1, -1], 'mask': [-1], 'file_name': []}, padding_values={
                'mfcc': -1., 'mask': 0., 'file_name': ''}).prefetch(tf.data.experimental.AUTOTUNE)