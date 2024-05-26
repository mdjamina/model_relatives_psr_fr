import argparse
import os
import re
import pandas as pd
from datasets import Dataset, Features, Sequence, Value, ClassLabel, DatasetDict
from nltk import RegexpTokenizer
from sklearn.model_selection import train_test_split


def load_excel_data(file_path: str, sheet_name: str) -> pd.DataFrame:
    """
    Load data from an Excel file.

    Args:
        file_path (str): The path to the Excel file.
        sheet_name (str): The name of the sheet to load.

    Returns:
        pd.DataFrame: The loaded data as a pandas DataFrame.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the sheet name does not exist in the Excel file.
    """
    # Check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    # Load the Excel file
    try:
        excel_file = pd.ExcelFile(file_path, engine='openpyxl')
    except Exception as e:
        raise ValueError(f"Error loading the Excel file: {e}")

    if sheet_name is not None:
        # Check if the sheet name exists in the Excel file
        if sheet_name not in excel_file.sheet_names:
            raise ValueError(
                f"The sheet {sheet_name} does not exist in the file {file_path}."
                f"Available sheets are: {excel_file.sheet_names}")

        # Load the data from the specified sheet
        try:
            df = pd.read_excel(excel_file, sheet_name=sheet_name, dtype=str)
        except Exception as e:
            raise ValueError(f"Error loading the data from sheet {sheet_name}: {e}")
    else:
        # Load the data from the first sheet
        try:
            df = pd.read_excel(excel_file, dtype=str)
        except Exception as e:
            raise ValueError(f"Error loading the data from the first sheet: {e}")
    # check if the required columns exist
    required_columns = {'texts', 'PSR', 'annotation'}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"The required columns {required_columns} are missing in the data."
                         f"Available columns are: {df.columns}")

    return df


def load_csv_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pd.DataFrame: The loaded data as a pandas DataFrame.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    # Check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    # Load the CSV file
    try:
        df = pd.read_csv(file_path, dtype=str)
    except Exception as e:
        raise ValueError(f"Error loading the CSV file: {e}")

    # check if the required columns exist
    required_columns = {'texts', 'PSR', 'annotation'}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"The required columns {required_columns} are missing in the data."
                         f"Available columns are: {df.columns}")

    return df


def load_data(file_path: str, sheet_name: str = None) -> pd.DataFrame:
    """
    Load data from an Excel or CSV file.

    Args:
        file_path (str): The path to the file.
        sheet_name (str): The name of the sheet to load. (default: None)

    Returns:
        pd.DataFrame: The loaded data as a pandas DataFrame.

    Raises:
        ValueError: If the file format is not supported.
    """

    # Check if the file exists
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    # Get the file extension
    file_ext = os.path.splitext(file_path)[1].lower()

    if file_ext in ['.xls', '.xlsx']:
        return load_excel_data(file_path, sheet_name)
    elif file_ext == '.csv':
        return load_csv_data(file_path)
    else:
        raise ValueError(f"The file format {file_ext} is not supported.")


def tokenizer(text):
    """
    Tokenize the text.

    Parameters:
    text (str): The text to tokenize.

    Returns:
    list of str: The list of tokens.
    """

    return RegexpTokenizer(r'''\w'|\w+|[^\w\s]''').tokenize(text)


def replace_characters(match: re.Match) -> str:
    """
    Replace special characters with their standard equivalents.

    Parameters:
    match (re.Match): The match object.

    Returns:
    str: The standard equivalent of the special character.
    """

    char = match.group(0)
    replacements = {
        '’': "'",
        '´': "'",
        '`': "'",
        '‘': "'",
        '«': '"',
        '»': '"',
        '“': '"',
        '”': '"',
        '–': '-',
        '—': '-',
        '…': ' ',
        u'\xa0': ' ',
    }

    return replacements[char]


def normalize_text(text: str) -> str:
    """
    Normalize the text by replacing special characters with their standard equivalents.

    Parameters:
    text (str): The text to normalize.

    Returns:
    str: The normalized text.
    """
    # Define the pattern to match the special characters
    pattern = r'[’´`‘«»“”–—…]'

    return re.sub(pattern, replace_characters, text).strip()


def annotate(text, psr, tag):
    """
    Annotate the text with labels indicating the presence of the first word of the exact sequence of words in the psr list.

    Parameters:
    text (list of str): The list of words to be annotated.
    psr (list of str): The sequence of words to be checked against.
    tag (str): The tag to annotate the text with.

    Returns:
    list of int: A list of labeled values where the label indicates the position of the psr sequence in the text.
    """
    # Initialize the annotation list with 0s
    annotation = [0] * len(text)

    # Define the labels
    labels = {'O': 0, 'det': 1, 'appo': 2, 'ambiguë': 3}

    # Find the start index of the subsequence psr in text
    psr_len = len(psr)
    text_len = len(text)

    for i in range(text_len - psr_len + 1):
        if text[i:i + psr_len] == psr:
            annotation[i] = labels[tag]
            break

    return annotation


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the data.

    Args:
        df (pd.DataFrame): The input data.

    Returns:
        pd.DataFrame: The preprocessed data.
    """
    # Drop the rows with missing values
    df = df.dropna()

    # Drop the duplicates
    df = df.drop_duplicates()

    # Normalize the text in dataframes
    df = df.map(normalize_text)

    # Tokenize the text in dataframes
    df['texts'] = df['texts'].apply(tokenizer)
    df['PSR'] = df['PSR'].apply(tokenizer)

    # Appliquer la fonction annotate à chaque ligne du DataFrame pour créer la colonne psr_tags
    df['psr_tags'] = df.apply(lambda row: annotate(row['texts'], row['PSR'], row['annotation']), axis=1)

    df.drop(columns=['PSR', 'annotation'], inplace=True)

    df.rename(columns={'texts': 'tokens'}, inplace=True)

    df.reset_index(names='id', inplace=True)

    return df


def create_datasets(df: pd.DataFrame, test_size: float = 0.2, valid_size: float = 0.2) -> DatasetDict:
    """
    Create the training, validation, and test datasets.

    Args:
        df (pd.DataFrame): The input data.
        test_size (float): The size of the test dataset.
        valid_size (float): The size of the validation dataset.

    Returns:
        DatasetDict: The training, validation, and test datasets.
    """
    # Split the data into training and testing sets
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)

    # Split the training data into training and validation sets
    train_df, valid_df = train_test_split(train_df, test_size=valid_size, random_state=42)

    # delete the index column
    train_df.reset_index(drop=True, inplace=True)
    valid_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)

    # Définir les Features pour chaque colonne du DataFrame
    features = Features({
        'id': Value(dtype='int64', id=None),
        'tokens': Sequence(feature=Value(dtype='string', id=None), length=-1, id=None),
        'psr_tags': Sequence(feature=ClassLabel(names=['O', 'DET', 'APPO', 'AMBIGUE'], id=None), length=-1, id=None)
    })

    # Convert the dataframes to datasets
    train_dataset = Dataset.from_pandas(train_df, features=features)
    valid_dataset = Dataset.from_pandas(valid_df, features=features)
    test_dataset = Dataset.from_pandas(test_df, features=features)

    # create datasetDict
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'valid': valid_dataset,
        'test': test_dataset
    })

    return dataset_dict


# upload the dataset to the Hugging Face Hub or to the local directory
def save_dataset(dataset: DatasetDict, name: str, directory: str = None, push_hub: bool = True):
    """
    Upload the dataset to the Hugging Face Hub.

    Args:
        dataset (DatasetDict): The dataset to upload.
        name (str): The name of the dataset.
        directory (str): The local directory to save the dataset. (if None, the dataset is uploaded to the Hugging Face Hub)
        push_hub (bool): Whether to upload the dataset to the Hugging Face Hub.
    """
    if directory:
        if directory[-1] != '/':
            directory += '/'
        dataset.save_to_disk(directory + name)

    if push_hub or not directory:
        dataset.push_to_hub(name)


if __name__ == '__main__':
    #dataset_name = 'relatives_psr_fr'
    #file_path = '../data/relatives.xlsx'
    #sheet_name = 'data'
    #local_dir = '../datasets/relatives_psr_fr'

    parser = argparse.ArgumentParser(description="Preprocess data")

    parser.add_argument('--file_path', type=str, default='../data/relatives.xlsx'
                        , help="The path to the Excel or CSV file."
                        , required=False)
    parser.add_argument('--sheet_name', type=str, default=None
                        , help="The name of the sheet to load."
                        , required=False)
    parser.add_argument('--dataset_name', type=str, default='relatives_psr_fr'
                        , help="The name of the dataset."
                        , required=False)
    parser.add_argument('--local_dir', type=str, default='../datasets/'
                        , help="The local directory to save the dataset."
                        , required=False)
    parser.add_argument('--push_hub', type=bool, default=True
                        , help="Whether to upload the dataset to the Hugging Face Hub."
                        , required=False)

    args = parser.parse_args()

    file_path = args.file_path
    sheet_name = args.sheet_name
    dataset_name = args.dataset_name
    local_dir = args.local_dir
    push_hub = args.push_hub

    print('Loading data...')
    data = load_data(file_path=file_path, sheet_name=sheet_name)
    # print(data.head())

    # Preprocess the data
    print('Preprocessing data...')
    preprocessed_data = preprocess_data(data)

    # Create the datasets
    print('Creating datasets...')
    datasets = create_datasets(preprocessed_data)

    # Upload the datasets to the Hugging Face Hub
    print('Uploading datasets...')
    save_dataset(datasets, name=dataset_name, directory=local_dir, push_hub=push_hub)

    print('Dataset uploaded successfully!')

    print('Done!')
