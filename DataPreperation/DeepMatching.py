import numpy as np
from pathlib import Path
from plistlib import load
from typing import Tuple, Dict, List, Callable
import re
import pandas as pd
import scipy.stats as stats
from sklearn.utils import resample


def read_siri_feature_and_label(passage_id: int,
                                siri_deep_feature_prefix: Path,
                                word_tagging_prefix: Path,
                                record_ids: List = None) -> List:
    """
    use passage id to locate the siri feature and label insert them into word tagging dict
    :param record_ids: a List that point out available siri index default 1 2 4 5
    :param passage_id: a passage id that is used to find the file
    :param siri_deep_feature_prefix: a prefix to the siri feature (npy) dir
    :param word_tagging_prefix: a prefix to the word tagging (plist) dir
    :return: List contains Dict that contains the feature and label
    """
    if record_ids is None:
        record_ids = [1, 2, 4, 5]
    # map all available siri record_ids to a path and concat with passage_id and read the plist file
    list_of_word_tagging_path = [word_tagging_prefix / f'SiriV{ids}' / f'{passage_id}.plist' for ids in record_ids]

    # open all the plist file and read the data
    list_of_word_tagging = [load(open(path, 'rb')) for path in list_of_word_tagging_path]
    # read the siri deep feature each npy contains all the siri deep feature for one passage
    siri_deep_feature = np.load(str(siri_deep_feature_prefix / f'{passage_id}.npy'))

    # list_of_word_tagging is a list of list, the outer list is different Siri record
    # the inner list is a list of dict in order of the word in the passage,
    # goal is to insert the siri deep feature for each word into the dict
    # the deep feature is array with length is padding length at the end of the passage since different siri record
    # has different length of segment
    # word_tagging_list is not padded, so the length of the list is the length of the passage
    # we need iterate through word_tagging_list and insert the deep feature into the dict
    for i, word_tagging_list in enumerate(list_of_word_tagging):
        for j, word_tagging_dict in enumerate(word_tagging_list):
            word_tagging_dict['siri_deep_feature'] = siri_deep_feature[i][j]
    return list_of_word_tagging


def parse_files_name(string: str) -> Dict:
    """
    Create a function that parse a string to a dict with sample
    the string has schema student_{student_id}_passage_{passage_id}_{random_number}

    Read a file name and return a dict with key as student_id, passage_id and random_number
    :param string: file name with schema student_{student_id}_passage_{passage_id}_{random_number}
    :return: dict with a key as student_id, passage_id and random_number
    """
    student_id = string.split('_')[1]
    passage_id = int(string.split('_')[3]) % 100000
    random_number = string.split('_')[4]
    return {'student_id': student_id, 'passage_id': passage_id, 'random_number': random_number}


def read_student_deep_feature(path_to_record: Path) -> Tuple:
    """
    read the student deep feature and recover passage id using name of the file
    :param path_to_record: path to the student deep feature npy file
    :return: tuple of passage id and student deep feature, and original file name
    """
    # read the file name
    file_name = str(path_to_record.name)
    # parse the file name to get the passage id
    passage_id = parse_files_name(file_name)['passage_id']
    # read the npy file
    student_deep_feature = np.load(str(path_to_record))
    return passage_id, student_deep_feature, file_name


def max_list_with_order(input_list: List[Tuple]) -> Tuple:
    """
    return the max element in the list, and the index of the max element in an original list
    :param input_list:  input for calculate max
    :return: (max_value, index)
    """
    # find the max element in the list_of_list
    max_value, max_ind = max(input_list, key=lambda x: x[0])
    return max_value, max_ind


def calculate_cosine_similarity_single(a: np.ndarray, b: np.ndarray):
    return np.dot(a / np.linalg.norm(a), b / np.linalg.norm(b))


def calculate_logits(x):
    # map -1 to 0 and 1 to 1
    p = (x + 1) / 2
    # compute the logit function
    logits = np.log(p / (1 - p))
    return logits


def calculate_cosine_similarity(a: np.ndarray, b: np.ndarray, threshold=-1., logits=False) -> Tuple[
    np.ndarray, np.ndarray]:
    # a, b are np.ndarray has the same shape in last dimension
    # normalize a and b on last dimension
    normed_a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    normed_b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    # calculate the cosine similarity in the last dimension the shape of the result is (a.shape[0], b.shape[0])
    cosine_similarity = np.matmul(normed_a, normed_b.T)
    # make linear the cosine similarity
    if logits:
        cosine_similarity = calculate_logits(cosine_similarity)
    # copy of the array for non modifying the original array
    cosine_similarity_ori = cosine_similarity.copy()
    # if the cosine similarity is smaller than a threshold, return -1 element wise
    cosine_similarity[cosine_similarity < threshold] = -1.
    return cosine_similarity, cosine_similarity_ori


def bootstrap_ci(df, column='difference', num_bootstrap_samples=10000):
    # Ensure there are no NaN values in the column
    score_diff = df[column].dropna()

    # Initialize lists to store bootstrap means and medians
    bootstrap_samples_mean = []
    bootstrap_samples_median = []

    # Perform bootstrap
    for _ in range(num_bootstrap_samples):
        bootstrap_sample = resample(score_diff)
        bootstrap_samples_mean.append(bootstrap_sample.mean())
        bootstrap_samples_median.append(bootstrap_sample.median())

    # Calculate 95% confidence interval for the mean
    ci_lower_mean = np.percentile(bootstrap_samples_mean, 2.5)
    ci_upper_mean = np.percentile(bootstrap_samples_mean, 97.5)

    # Calculate 95% confidence interval for the median
    ci_lower_median = np.percentile(bootstrap_samples_median, 2.5)
    ci_upper_median = np.percentile(bootstrap_samples_median, 97.5)

    print(f'Bootstrap 95% confidence interval for the mean: ({ci_lower_mean:.2f}, {ci_upper_mean:.2f})')
    print(f'Bootstrap 95% confidence interval for the median: ({ci_lower_median:.2f}, {ci_upper_median:.2f})')


def analyze_score_diff(df, column='difference', alpha=0.0015):
    score_diff = df[column].dropna()

    # Perform Wilcoxon signed-rank test against the hypothesized median = 0
    st_rt = stats.wilcoxon(score_diff)
    p_value = st_rt.pvalue
    print("Performing a Wilcoxon signed-ranked test.")
    print("The null hypothesis of this test is that the median of the differences is zero.")
    print(f'Test statistic (W): {st_rt.statistic}')
    print(f'P-value: {p_value}')

    # Conclusion
    if p_value < alpha:
        print(f"The p-value is less than {alpha}. Therefore, we reject the null hypothesis. This means that "
              f"if the true median of the differences were zero, the probability of observing a median as extreme as, "
              f"or more extreme than, the one in our data by chance would be approximately {p_value * 100 :.2f}%.")
    else:
        print(
            f"The p-value is greater than {alpha}. Therefore, we fail to reject the null hypothesis. This suggests that "
            f"if the true median of the differences were zero, we would expect to see a median as extreme as, or more "
            f"extreme than, the one in our data by chance with a probability of approximately {p_value * 100 :.2f}%.")

    bootstrap_ci(df, column=column)


def sequence_matching(student_deep_feature: np.ndarray,
                      list_of_word_tagging: List[Dict],
                      threshold: float = 0.8,
                      similarity_function: Callable[[np.ndarray, np.ndarray, float], np.ndarray]
                      = calculate_cosine_similarity) -> Tuple:
    """
    Using Modified Longest Common Subsequence to find the matching sequence, using cosine similarity to measure if
    it is a match, if it is not a match, similarity_function will return -1, if it is match, similarity_function will
    return a number between the threshold and 1. The goal is to find the longest sequence that has the highest sum of
    similarity_function return value. The function will return a list which is matched list_of_word_tagging(removed
    non-matched word_tagging_dict) and a list of index that matched in student_deep_feature.

    The transfer function is: dp[x][y]=max( dp[x][y-1], dp[x-1][y], sim(x,y) + dp[x-1][y-1] );
    the max_list_with_order return a tuple of (max_value, index) we can need to add index to the tuple before make it a
    list is use for find the back pointer


    :param similarity_function: a function that takes two np.ndarray and a threshold return similarity score
    :param student_deep_feature: a np.ndarray that contains the student deep feature shape in
    (number of segments, deep feature dim)
    :param list_of_word_tagging: a list of single siri record in format of dict, each dict contains siri_deep_feature
     that is for calculate similarity
    :param threshold: an int that is used to filter the similarity score
    :return: list of matched list_of_word_tagging and list of index that matched in student_deep_feature
    """
    # get all deep features from list_of_word_tagging
    ref_deep = np.array(list(map(lambda x: x['siri_deep_feature'], list_of_word_tagging)))
    # calculate similarity score for batch
    overall_similarity_scores, cosine_similarity_ori = similarity_function(student_deep_feature, ref_deep, threshold)
    # initialize the dp table
    dp_table = np.zeros((len(student_deep_feature) + 1, len(list_of_word_tagging) + 1))
    # initialize the back pointer table using dict for easy access
    back_pointer_table = {}
    # start matching
    for matrix_index_i in range(1, np.shape(dp_table)[0]):
        student_index = matrix_index_i - 1
        for matrix_index_j in range(1, np.shape(dp_table)[1]):
            reference_index = matrix_index_j - 1
            # calculate the similarity score
            similarity_score = overall_similarity_scores[student_index][reference_index]
            # update the dp table in matching case and non-matching case
            # since if a non-matching case, the similarity score will be -1, so we can just use max function to update
            # get max value and it's source
            calculate_list = [(dp_table[matrix_index_i - 1][matrix_index_j], (matrix_index_i - 1, matrix_index_j)),
                              (dp_table[matrix_index_i][matrix_index_j - 1], (matrix_index_i, matrix_index_j - 1)),
                              (dp_table[matrix_index_i - 1][matrix_index_j - 1] + similarity_score,
                               (matrix_index_i - 1, matrix_index_j - 1))]
            max_value, max_index = max_list_with_order(calculate_list)
            # update the dp table
            dp_table[matrix_index_i][matrix_index_j] = max_value
            # update the back pointer table
            back_pointer_table[(matrix_index_i, matrix_index_j)] = max_index
    # find the max value in the dp table and back trace to find the matched sequence
    # find the max value index in the dp_table
    max_value_indices = np.unravel_index(dp_table.argmax(), dp_table.shape)
    current_index = max_value_indices
    matched_word_tagging = []
    matched_indices = []
    matched_similarity_score = []
    while current_index in back_pointer_table:
        previous_index = back_pointer_table[current_index]

        # Check if we moved diagonally. If we did, it means we have a match.
        if current_index[0] - 1 == previous_index[0] and current_index[1] - 1 == previous_index[1]:
            # Append to matched lists (prepend because we are going in reverse order)
            matched_word_tagging.insert(0, list_of_word_tagging[previous_index[1]])
            matched_indices.insert(0, previous_index[0])
            matched_similarity_score.insert(0, overall_similarity_scores[previous_index[0]][previous_index[1]])
        # Move to the previous cell
        current_index = previous_index
    fitness = dp_table[max_value_indices]
    return matched_word_tagging, matched_indices, matched_similarity_score, fitness, cosine_similarity_ori


def get_information_for_all(search_dir: Path,
                            siri_deep_feature_prefix: Path,
                            word_tagging_prefix: Path,
                            csv_result: Path) -> pd.DataFrame:
    processed_case = []
    new_matched_adj_count = []
    old_matched_adj_count = []
    df_old = pd.read_csv(csv_result, index_col=0, compression='gzip')
    for record in search_dir.glob('*.npy'):
        case_id = record.stem
        # tqdm.set_postfix({"Current case_id:": case_id}, refresh=True)
        passage_id, student_deep_feature, file_name = read_student_deep_feature(record)
        list_of_word_tagging = read_siri_feature_and_label(passage_id, siri_deep_feature_prefix, word_tagging_prefix)
        try:
            old_string = df_old.loc[case_id]['result']
            old_matched_string = ' '.join(map(lambda x: x.strip(),
                                              re.findall(r'([a-zA-Z\s]+)(?=<0\.\d{2}>)', old_string))).split(' ')
            # remove all empty strings
            old_matched_string = list(filter(lambda x: x.strip() != '', old_matched_string))
            # print(f'{case_id} found')
        except KeyError:
            print(f'{case_id} not found')
            old_matched_string = []
        max_fitness = 0
        max_adjusted_matched_count = 0
        for i in range(len(list_of_word_tagging)):
            matched_word_tagging, \
                matched_indices, \
                matched_similarity_score, \
                fitness_score, ori \
                = sequence_matching(student_deep_feature[0], list_of_word_tagging[i], 0.0)
            matched_string = ' '.join(list(map(lambda x: x['tString'], matched_word_tagging)))
            if fitness_score > max_fitness:
                max_fitness = fitness_score
                max_id = i
                max_adjusted_matched_count = len(matched_string.split(' '))
        processed_case.append(case_id)
        new_matched_adj_count.append(max_adjusted_matched_count)
        old_matched_adj_count.append(len(old_matched_string))
    # create a dataframe using processed as index, and new_matched_adj_count, old_matched_adj_count as columns
    df = pd.DataFrame({'new_matched_adj_count': new_matched_adj_count,
                       'old_matched_adj_count': old_matched_adj_count}, index=processed_case)
    # calculate the difference between new_matched_adj_count and old_matched_adj_count
    df['difference'] = df['new_matched_adj_count'] - df['old_matched_adj_count']
    return df


# %%
# Run the function
model_name = "Deep_Feature_model_7_Numpy"
search_dir = Path(f"../DataFolder/Student_Response/{model_name}")
siri_deep_feature_prefix = Path(f"../DataFolder/Siri_Related/{model_name}")
word_tagging_prefix = Path("../DataFolder/Siri_Related/SiriR")
old_path_prefix = Path("../DataFolder/Student_Response/Match/result.csv.gz")
df = get_information_for_all(search_dir, siri_deep_feature_prefix, word_tagging_prefix, old_path_prefix)
df2 = df[df['old_matched_adj_count'] != 0]

# score_diff is a column 'difference' in df2 DataFrame
analyze_score_diff(df2)

#%%
# Test Cell
case_id = 'student_982_passage_34000_553fe870c3878'
siri_deep_feature_prefix = Path(f"../DataFolder/Siri_Related/{model_name}")
word_tagging_prefix = Path("../DataFolder/Siri_Related/SiriR")
single_result = search_dir / f'{case_id}.npy'
# Previous matching result
df_old = pd.read_csv(old_path_prefix, index_col=0)
old_string = df_old.loc[case_id]['result']
old_matched_string = ' '.join(map(lambda x: x.strip(), re.findall(r'([a-zA-Z\s]+)(?=<0\.\d{2}>)', old_string))).split(' ')
# read the file name
passage_id, student_deep_feature, file_name = read_student_deep_feature(single_result)
# read the siri file
list_of_word_tagging = read_siri_feature_and_label(passage_id, siri_deep_feature_prefix, word_tagging_prefix)
# calculate the matching
# generate all the matching
max_fitness = 0
for i in range(len(list_of_word_tagging)):
    matched_word_tagging, \
        matched_indices, \
        matched_similarity_score, \
        fitness_score, ori\
        = sequence_matching(student_deep_feature[0], list_of_word_tagging[i], 0.0)
    matched_string = ' '.join(list(map(lambda x: x['tString'], matched_word_tagging)))
    # print Siri id and matched string and fitness score (only keep 2 decimal)
    ref_string = ' '.join(list(map(lambda x: x['tString'], list_of_word_tagging[i])))
    print(f"Siri id: {i}, \n"
          f"Fitness Score {fitness_score:.2f}, "
          f"Matched Count: {len(matched_word_tagging)}, "
          f"Reference Count:{len(list_of_word_tagging[i])}, "
          f"Adjusted Matched Count: {len(matched_string.split(' '))}, "
          f"Adjusted Reference Count: {len(ref_string.split(' '))},\n "
          f"matched string: \n\t{matched_string}", end='\n\n')
    if fitness_score > max_fitness:
        # update the max fitness score
        max_fitness = fitness_score
        # save Adjusted Matched Count and Adjusted Reference Count's ID
        max_fitness_id = i
        max_adjusted_matched_count = len(matched_string.split(' '))
# Print Reference String
ref_string = ' '.join(list(map(lambda x: x['tString'], list_of_word_tagging[max_fitness_id])))
print(f"Student Response Count: {student_deep_feature[0].shape[0]},\t"
      f"Previous Matched Count: {len(old_matched_string)},\t"
      f"Current Matched Count: {max_adjusted_matched_count},\n"
      f"Reference String: \n\t{ref_string}")


# %%
