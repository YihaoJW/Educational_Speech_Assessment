import json
import string
from typing import List, Dict, Tuple

import nltk


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
        self.__metrics = None
        str_set = "".join([char for char in string.punctuation if char != "'"])
        self.__translator = str.maketrans("", "", str_set)

    @staticmethod
    def __create_passage_dict(json_path: str) -> Dict:
        """
        Read a json file and return a dict with key as passage id and value as passage
        :param json_path: path to reference passage csv file
        :return: a dict contains passage information
        """
        passage_dict = json.load(open(json_path))
        return passage_dict

    def __tokenize_text(self, text: str) -> List[Dict]:
        """
        Read a text and return a list of tokenized text
        :param text: text to be tokenized
        :return: a list of tokenized text
        """
        # tokens = nltk.word_tokenize(text) # Causing problem with the tokenization
        # remove punctuation from the text
        tokens = text.translate(self.__translator).split()

        return [{"content": token.lower(), "confidence": -1} for token in tokens if token != "" and token != " "]

    def __query_ref_text(self, passage_id: int) -> str:
        """
        Read a passage_id and return the reference text
        :param passage_id: passage id
        :return: reference text
        """
        return self.__passage_dict[str(passage_id)]

    def read_student_text(self, json_bin: bytes, student_id: str, passage_id: int, session_id: int) -> List[Dict]:
        """
        Read a student text and return a list of tokenized text
        :param session_id:
        :param passage_id:
        :param student_id:
        :param json_bin:
        :return: a list of tokenized text
        """

        def content_mapper(x):
            content = x["alternatives"][0]["content"]
            confidence = float(x["alternatives"][0]["confidence"])
            start_time = float(x["start_time"])
            end_time = float(x["end_time"])
            return {"content": content, "confidence": confidence, "start_time": start_time, "end_time": end_time}

        data = json.loads(json_bin)
        data = [content_mapper(x) for x in data['results']['items'] if x['type'] == 'pronunciation']
        self.__case_data = data
        self.__file_name = session_id
        self.__case_id = {'student_id': student_id, 'passage_id': passage_id, 'session_id': session_id}
        self.__ref_original_text = self.__tokenize_text(self.__query_ref_text(self.__case_id['passage_id']))
        if self.__case_id['passage_id'] not in self.__stemmed_dict:
            self.__stemmed_dict[self.__case_id['passage_id']] = self.__stem(self.__ref_original_text)
        self.__ref_text = self.__stemmed_dict[self.__case_id['passage_id']]
        self.__result = None
        self.__metrics = None
        return data

    def __stem(self, text: List[Dict]) -> List[Dict]:
        """
        Internal Stem mapping function
        if stem is enabled, it will stem the text else it will return the original text
        to increase the speed of the algorithm, it will only stem the text once and store it in a dict self.__stemmed_dict
        :param text: List of Dict with key as content and confidence
        :return: List of Dict with key as content and confidence
        """
        if self.stemming:
            if self.__case_id['passage_id'] not in self.__stemmed_dict:
                self.__stemmed_dict[self.__case_id['passage_id']] = \
                    [{"content": self.stemmer.stem(token['content']), "confidence": token['confidence']}
                     for token in text]
            return self.__stemmed_dict[self.__case_id['passage_id']]
        else:
            return text

    def lcss(self) -> str:
        """
        A function take two lists of words in sequence as input and
        return using The Longest Common Subsequence algorithm
        :return: a list of tuple of (dict, status) where dict is a dict with key contain as
        content and confidence
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
            x = (rec_x['content']).lower()
            for j, rec_y in enumerate(student_input):
                y = (stem(rec_y['content'])).lower()
                if x == y and float(rec_y['confidence']) >= self.__match_prob:
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
                assert ref_text[x - 1]['content'].lower() == stem(student_input[y - 1]['content']).lower()
                result.append((student_input[y - 1], "M"))
                x -= 1
                y -= 1
                self.__result = result[::-1]
        return self.__get_string_with_status(self.__result)

    @staticmethod
    def __get_string_with_status(lcs_matching_return: List[Tuple[Dict, str]]) -> str:
        """
        A function that input a list of tuple of (dict, tuple) the dict must contain key as content and confidence
        :param lcs_matching_return: a list of tuple of (dict, tuple) the dict must contain key as content and
        confidence
        :return: a string of word with status
        """
        common_tuple = [(x['content'], float(x['confidence']), c) for x, c in lcs_matching_return]
        return ' '.join(
            [f'{word}<{status}>' if status != 'M' else word if confidence > 0.9 else f'{word}<{confidence:.2f}>' for
             word, confidence, status in common_tuple]
        )

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
        :return: List of Dict with key as content and confidence
        """
        return self.__ref_text

    def get_case_data(self) -> List[Dict]:
        """
        A function that return the case data
        :return: List of Dict with key as content and confidence
        """
        return self.__case_data

    def get_file_name(self) -> str:
        """
        A function that return the file name
        :return: string of file name
        """
        return self.__file_name

    def calculate_metrics(self) -> Dict:
        """
        Calculate the metrics for the given case data and reference text.
        :return: a dictionary containing the metrics
        """
        if self.__metrics is not None:
            return self.__metrics

        if self.__result is None:
            self.lcss()
        total_student_words = len([x for x in self.__result if x[1] != 'R'])
        correctly_spoken_words = sum([1 for x in self.__result if x[1] == 'M'])
        correct_rate = sum([1 for x in self.__result if x[1] == 'M']) / total_student_words
        # Time elapsed from first word to last word
        time_diff = self.__case_data[-1]['end_time'] - self.__case_data[0]['start_time']
        self.__metrics = {
            "correct_rate": correct_rate,
            "total_student_words": total_student_words,
            "correctly_spoken_words": correctly_spoken_words,
            "time_elapsed": time_diff,
            "wcpm": correctly_spoken_words / time_diff * 60
        }
        return self.__metrics

    def nice_print(self) -> str:
        """
        Print the result in a nice format to either console or file specified.

        :param string:
        :param file: The file path where the output should be written. If None, prints to console.
        """
        output = []
        if self.__result is None:
            result = self.lcss()  # Ensure that the computation is done before printing
        else:
            result = self.__get_string_with_status(self.__result)

        # Collect output data
        output.append("Student ID: {}".format(self.__case_id['student_id']))
        output.append("Passage ID: {}".format(self.__case_id['passage_id']))
        output.append("Session ID: {}".format(self.__case_id['session_id']))
        output.append("Reference Text:\n\t {}\n".format(' '.join([x['content'] for x in self.__ref_original_text])))
        output.append("Student Text:\n\t {}\n".format(' '.join([x['content'] for x in self.__case_data])))
        if self.__result is not None:
            output.append("Result:\n\t {}\n".format(result))
            metrics = self.calculate_metrics()
            output.append("Correct Rate: {:.2f}%\t\t"
                          "Correctly/Word Spoken: {}/{}\t\t"
                          "Time Elapsed: {:.2f}\t\t"
                          "WCPM: {:.2f}\n".format(metrics['correct_rate'] * 100,
                                                  metrics['correctly_spoken_words'],
                                                  metrics['total_student_words'],
                                                  metrics['time_elapsed'],
                                                  metrics['wcpm']))
        # Convert list to single string
        output_string = "\n".join(output) + "\n"

        return output_string


if __name__ == '__main__':
    print('Script running')
    import sys

    print(sys.version)

#%%
