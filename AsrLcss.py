import plistlib
from typing import List, Dict, Tuple

import nltk
import pandas as pd


class AsrLcss:
    def __init__(self, dict_path: str = None,
                 passage_dict: Dict = None,
                 match_prob: float = 0.1,
                 stemmer: nltk.stem.api = None):
        """
        :param dict_path: path to the dictionary
        :param passage_dict: a dict of passage id and passage
        :param match_prob: a float number between 0 and 1 that represent the minimum confidence
        :param stemmer: a stemmer object
        """
        if dict_path is not None:
            self.__passage_dict = self.__create_passage_dict(dict_path)
        else:
            assert passage_dict is not None
            self.__passage_dict = passage_dict
        self.__stemmed_dict = {}
        self.__match_prob = match_prob
        if stemmer is not None:
            self.stemmer = stemmer
            self.stemming = True
        else:
            self.stemming = False
            self.stemmer = None
        self.__case_data = None
        self.__case_id = None
        self.__file_name = None
        self.__ref_text = None
        self.__ref_original_text = None
        self.__match_prob = match_prob
        self.__result = None

    @staticmethod
    def __create_passage_dict(csv_path: str) -> Dict:
        """
        Read a csv file and return a dict with key as passage id and value as passage
        :param csv_path: path to reference passage csv file
        :return: a dict contains passage information
        """
        df = pd.read_csv(csv_path)
        passage_dict = {row['passage_id']: row['passage'] for _, row in df.iterrows()}
        return passage_dict

    @staticmethod
    def __parse_files_name(file_name: str) -> Dict[str, int]:
        """
        Read a file name and return a dict with key as student_id, passage_id and session_id
        :param file_name: file name with schema student_{student_id}_passage_{passage_id}_{session_id}
        :return: dict with key as student_id, passage_id and session_id
        """
        student_id = file_name.split('_')[1]
        passage_id = int(file_name.split('_')[3])
        session_id = file_name.split('_')[4]
        return {'student_id': student_id, 'passage_id': passage_id, 'session_id': session_id}

    @staticmethod
    def __tokenize_text(text: str) -> List[Dict]:
        """
        Read a text and return a list of tokenized text
        :param text: text to be tokenized
        :return: a list of tokenized text
        """
        tokens = nltk.word_tokenize(text)
        return [{"tString": token.lower(), "tConfidence": -1} for token in tokens if token.isalpha()]

    def __query_ref_text(self, passage_id: int) -> str:
        """
        Read a passage_id and return the reference text
        :param passage_id: passage id
        :return: reference text
        """
        return self.__passage_dict[passage_id]

    def read_student_text(self, path_to_plist: str) -> List[Dict]:
        """
        Read a student text and return a list of tokenized text
        :param path_to_plist:
        :return: a list of tokenized text
        """
        with open(path_to_plist, 'rb') as f:
            data = plistlib.load(f)
        self.__case_data = data
        self.__file_name = path_to_plist.split('/')[-1].split('.')[0]
        self.__case_id = self.__parse_files_name(self.__file_name)
        self.__ref_original_text = self.__tokenize_text(self.__query_ref_text(self.__case_id['passage_id']))
        if self.__case_id['passage_id'] not in self.__stemmed_dict:
            self.__stemmed_dict[self.__case_id['passage_id']] = self.__stem(self.__ref_original_text)
        self.__ref_text = self.__stemmed_dict[self.__case_id['passage_id']]
        self.__result = None
        return data

    def __stem(self, text: List[Dict]) -> List[Dict]:
        """
        Internal Stem mapping function
        if stem is enabled, it will stem the text else it will return the original text
        to increase the speed of the algorithm, it will only stem the text once and store it in a dict self.__stemmed_dict
        :param text: List of Dict with key as tString and tConfidence
        :return: List of Dict with key as tString and tConfidence
        """
        if self.stemming:
            if self.__case_id['passage_id'] not in self.__stemmed_dict:
                self.__stemmed_dict[self.__case_id['passage_id']] = \
                    [{"tString": self.stemmer.stem(token['tString']), "tConfidence": token['tConfidence']}
                     for token in text]
            return self.__stemmed_dict[self.__case_id['passage_id']]
        else:
            return text

    def lcss(self) -> str:
        """
        A function take two lists of words in sequence as input and
        return using The Longest Common Subsequence algorithm
        :return: a list of tuple of (dict, status) where dict is a dict with key contain as
        tString and tConfidence
        """
        student_input = self.__case_data
        ref_text = self.__ref_text
        stemming = self.stemming
        if stemming:
            stem = self.stemmer.stem
        else:
            stem = lambda xa: xa
        lengths = [[0 for j in range(len(student_input) + 1)] for i in range(len(ref_text) + 1)]
        # row 0 and column 0 are initialized to 0 already
        for i, rec_x in enumerate(ref_text):
            x = (rec_x['tString']).lower()
            for j, rec_y in enumerate(student_input):
                y = (stem(rec_y['tString'])).lower()
                if x == y and rec_y['tConfidence'] >= self.__match_prob:
                    lengths[i + 1][j + 1] = lengths[i][j] + 1
                else:
                    lengths[i + 1][j + 1] = max(lengths[i + 1][j], lengths[i][j + 1])
        # read the substring out from the matrix
        result = []
        x, y = len(ref_text), len(student_input)
        while x != 0 and y != 0:
            if lengths[x][y] == lengths[x - 1][y]:
                result.append((self.__ref_original_text[x - 1], "R"))
                x -= 1
            elif lengths[x][y] == lengths[x][y - 1]:
                result.append((student_input[y - 1], "A"))
                y -= 1
            else:
                assert ref_text[x - 1]['tString'].lower() == stem(student_input[y - 1]['tString']).lower()
                result.append((student_input[y - 1], "M"))
                x -= 1
                y -= 1
                self.__result = result[::-1]
        return self.__get_string_with_status(self.__result)

    @staticmethod
    def __get_string_with_status(lcs_matching_return: List[Tuple[Dict, str]]) -> str:
        """
        A function that input a list of tuple of (dict, tuple) the dict must contain key as tString and tConfidence
        :param lcs_matching_return: a list of tuple of (dict, tuple) the dict must contain key as tString and
        tConfidence
        :return: a string of word with status
        """
        common_tuple = [(x['tString'], x['tConfidence'], c) for x, c in lcs_matching_return]
        return ' '.join(
            [f'{word}<{status}>' if status != 'M' else f'{word}<{confidence:.2f}>' for word, confidence, status in
             common_tuple])

    def get_result(self) -> List[Tuple[Dict, str]]:
        """
        A function that return the result of lcss
        :return:
        """
        return self.__result

    def get_case_id(self) -> Dict:
        """
        A function that return the case id
        :return: a dict with key as student_id, passage_id and session_id
        """
        return self.__case_id

    def get_ref_text(self) -> List[Dict]:
        """
        A function that return the reference text
        :return: List of Dict with key as tString and tConfidence
        """
        return self.__ref_text

    def get_case_data(self) -> List[Dict]:
        """
        A function that return the case data
        :return: List of Dict with key as tString and tConfidence
        """
        return self.__case_data

    def get_file_name(self) -> str:
        """
        A function that return the file name
        :return: string of file name
        """
        return self.__file_name

    def nice_print(self) -> None:
        """
        Print the result in a nice format
        """
        if self.__result is None:
            self.__result = self.lcss()
        print("Student ID: {}".format(self.__case_id['student_id']))
        print("Passage ID: {}".format(self.__case_id['passage_id']))
        print("Session ID: {}".format(self.__case_id['session_id']))
        print("Reference Text:\n\t {}\n".format(' '.join([x['tString'] for x in self.__ref_original_text])))
        print("Student Text:\n\t {}\n".format(' '.join([x['tString'] for x in self.__case_data])))
        if self.__result is not None:
            print("Result:\n\t {}\n".format(self.__result))
        print("")


if __name__ == '__main__':
    print('Script running')
    import sys

    print(sys.version)
