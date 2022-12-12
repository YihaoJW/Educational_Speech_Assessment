import plistlib
from typing import List, Dict, Tuple, Iterator, Iterable

import nltk
import pandas as pd


# %% A function that read a path of *.plist file and return a list of dictionaries and the name of the file without
# the extension
def read_plist(path: str) -> Tuple[List[Dict], str]:
    """
    Read a plist file and return a list of dictionaries and the name of the file without the extension

    :param path:
    Path str to the plist file

    :return: a tuple of (list of dictionaries, file name) that dictionary is word information that student read
    """
    with open(path, 'rb') as f:
        data = plistlib.load(f)
    return data, path.split('/')[-1].split('.')[0]


# %%
# create a dict with key as passage id and value as passage using information in the csv files
def create_passage_dict(csv_path: str) -> Dict:
    """
    Read a csv file and return a dict with key as passage id and value as passage
    :param csv_path: path to reference passage csv file
    :return: a dict contains passage information
    """
    df = pd.read_csv(csv_path)
    passage_dict = {row['passage_id']: row['passage'] for _, row in df.iterrows()}
    return passage_dict


# %%
# create a function that parse a string to a dict with sample
# the string has schema student_{student_id}_passage_{passage_id}_{random_number}
def parse_files_name(string: str) -> Dict:
    """
    Read a file name and return a dict with key as student_id, passage_id and random_number
    :param string: file name with schema student_{student_id}_passage_{passage_id}_{random_number}
    :return: dict with key as student_id, passage_id and random_number
    """
    student_id = string.split('_')[1]
    passage_id = int(string.split('_')[3]) % 100000
    random_number = string.split('_')[4]
    return {'student_id': student_id, 'passage_id': passage_id, 'random_number': random_number}


# %% a function read a string defined in schema student_{student_id}_passage_{passage_id}_{random_number} and return
# tokenized text remove stop words and punctuation
def read_and_tokenize_file(file_name: str, passage_dict: Dict) -> List[Dict]:
    """
    read a file name and search related passage in the passage_dict and return a dict that tokenized text with
    remove stop words and punctuation
    :param file_name: file name with schema student_{student_id}_passage_{passage_id}_{random_number}
    :param passage_dict: passage dict with key as passage_id and value as passage
    :return: a dict that tokenized text remove stop words and punctuation
    """
    idx = parse_files_name(file_name)['passage_id']
    try:
        textx = passage_dict[idx]
    except KeyError:
        try:
            textx = passage_dict[int(str(idx)[1:])]
        except KeyError:
            raise KeyError
    tokens = nltk.word_tokenize(textx)
    return [{"tString": token.lower(), "tConfidence": -1} for token in tokens if token.isalpha()]


# %%
# A function that input an iterator return the first item that is no error
# the iterator contains a tuple of (data, file_name)
# map the file name with to a passage using read_and_tokenize_file
# return a tuple of ((tokenized_text, data), file_name)
# if there is a KeyError, it will return the next item
def get_next_item(it: Iterator, passage_dict: Dict) -> Tuple[Tuple[List[Dict], List[Dict]], str]:
    """
    A function that input an iterator return the first item that is no error
    :param it: a iterator contains a tuple of (data, file_name)
    :param passage_dict: passage dict with key as passage_id and value as passage
    :return: a tuple of ((tokenized_text, data), file_name)
    """
    while True:
        try:
            data, file_name = next(it)
            tokenized_text = read_and_tokenize_file(file_name, passage_dict)
            return (tokenized_text, data), file_name
        except KeyError:
            continue


# %%
# A function take two lists of words in sequence as input and return using The Longest Common Subsequence algorithm
def lcss(ref_text: List[Dict], student_input: List[Dict], prob: float = 0.1, stemmer: nltk.stem.api = None) -> \
        List[Tuple[Dict, str]]:
    """
    A function take two lists of words in sequence as input and return using The Longest Common Subsequence algorithm

    :param ref_text: a list of words in sequence that each element is a dict with key must contain as tString and
    tConfidence

    :param student_input: a list of words in sequence that each element is a dict with key must contain
    as tString and tConfidence

    :param stemmer: a stemmer object from nltk.Stem
    :param prob: a float number that is the threshold of confidence

    :return: a list of tuple of (dict, status) where dict is a dict with key contain as
    tString and tConfidence
    """
    stemming = stemmer is not None
    lengths = [[0 for j in range(len(student_input) + 1)] for i in range(len(ref_text) + 1)]
    # row 0 and column 0 are initialized to 0 already
    for i, rec_x in enumerate(ref_text):
        x = (stemmer.stem(rec_x['tString']) if stemming else rec_x['tString']).lower()
        for j, rec_y in enumerate(student_input):
            y = (stemmer.stem(rec_y['tString']) if stemming else rec_y['tString']).lower()
            if x == y and rec_y['tConfidence'] >= prob:
                lengths[i + 1][j + 1] = lengths[i][j] + 1
            else:
                lengths[i + 1][j + 1] = max(lengths[i + 1][j], lengths[i][j + 1])
    # read the substring out from the matrix
    result = []
    x, y = len(ref_text), len(student_input)
    while x != 0 and y != 0:
        if lengths[x][y] == lengths[x - 1][y]:
            result.append((ref_text[x - 1], "R"))
            x -= 1
        elif lengths[x][y] == lengths[x][y - 1]:
            result.append((student_input[y - 1], "A"))
            y -= 1
        else:
            if not stemming:
                assert ref_text[x - 1]['tString'] == student_input[y - 1]['tString'].lower()
            else:
                assert stemmer.stem(ref_text[x - 1]['tString'].lower()) \
                       == stemmer.stem(student_input[y - 1]['tString'].lower())
            result.append((student_input[y - 1], "M"))
            x -= 1
            y -= 1
    return result[::-1]


# %%
# A function that input a list of tuple of (dict, tuple) and return a string of word with status
# return confidence with 2 digits if the status is M otherwise add the status as a tag at the end of the word
def get_string_with_status(lcs_matching_return: List[Tuple[Dict, str]]) -> str:
    """
    A function that input a list of tuple of (dict, tuple) the dict must contain key as tString and tConfidence
    :param lcs_matching_return: a list of tuple of (dict, tuple) the dict must contain key as tString and tConfidence
    :return: a string of word with status
    """
    common_tuple = [(x['tString'], x['tConfidence'], c) for x, c in lcs_matching_return]
    return ' '.join(
        [f'{word}<{status}>' if status != 'M' else f'{word}<{confidence:.2f}>' for word, confidence, status in
         common_tuple])


# %% iterate over all files in file_list and save the result by file name under a new folder Match with extension
# .txt and content is get_string_with_status(common_tuple) if there is a KeyError, it will skip the file overwrite
# the file if it already exists

def save_match_result(file_list: Iterable[str], passage_dict: Dict[int, str]) -> None:
    """
    iterate over all files in file_list and save the result by file name under a new folder Match with extension
    :param file_list: an iterable contain path to file
    :param passage_dict: a dict with key as passage_id and value as passage
    """
    for file_path in file_list:
        try:
            data, file_name = read_plist(file_path)
            tokenized_text = read_and_tokenize_file(file_name, passage_dict)
            common = lcss(tokenized_text, data)
            common_tuple = [(x['tString'], x['tConfidence'], c) for x, c in common]
            with open(f'DataFolder/Student_Response/Match/{file_name}.txt', 'w') as f:
                f.write(get_string_with_status(common_tuple))
        except KeyError:
            continue


# %% iterate over all files in file_list and save the result by csv table with index as file name and content is
# get_string_with_status(common_tuple) if there is a KeyError, it will skip the file sort the table by file name
# compress with gzip and overwrite the file if it already exists
def save_result_csv(file_list: Iterable[str], passage_dict: Dict[int, str]) -> None:
    """
    iterate over all files and get result of Longest Common Subsequence and save the result by csv table with index as file name
    :param file_list: an iterable contain path to file
    :param passage_dict: a dict with key as passage_id and value as passage
    """
    result = []
    for file_path in file_list:
        try:
            data, file_name = read_plist(file_path)
            tokenized_text = read_and_tokenize_file(file_name, passage_dict)
            common = lcss(tokenized_text, data)
            common_tuple = [(x['tString'], x['tConfidence'], c) for x, c in common]
            result.append((file_name, get_string_with_status(common_tuple)))
        except KeyError:
            continue
    df = pd.DataFrame(result, columns=['file_name', 'result'])
    df = df.sort_values(by=['file_name'])
    df.to_csv('result.csv.gz', index=False, compression='gzip')
