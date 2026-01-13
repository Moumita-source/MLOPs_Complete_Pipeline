import os
import logging
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import spacy
nlp = spacy.load('en_core_web_sm')


# Ensure the "logs" directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok= True)

# Setting up logger
logger = logging.getLogger('data_preprocessing')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'data_preprocessing.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def remove_stopwords_and_lemmatize(text):

    """
    Transforms the text after removing stop words and lemmmatizing
    """

    doc = nlp(text)
    ## Collect lematize words after removing stop words
    words = [token.lemma_ for token in doc if not token.is_stop]
    ## Join the words
    return " ".join(words)

def transform_text(series):
    """ 
    Transforms the input text by converting it to lowercase, tokenizing, removing stop words and punctuations and stemming
    """
    # Convert to lowercase
    series = series.str.lower()
    ## Keeping only the characters, removing the special characters, punctuations etc
    series = series.replace(r'[^A-Za-z\s]', '', regex = True)
    ## Normalize space
    series = series.replace(r'\s+', ' ', regex = True)
    ## Remove any trailling spaces
    series = series.str.strip()
    ## Removing stopwords and lemmatizing
    series = series.apply(remove_stopwords_and_lemmatize)

    return series

def preprocess_df(df, text_column = 'text', target_column = 'target'):
    """
    Preprocess the DataFrame by encoding the target column, removing duplicates, and transforming the text column
    """

    try:
        logger.debug('Starting preprocessing for DataFrame')
        # Encode the target column
        encoder = LabelEncoder()
        df[target_column] = encoder.fit_transform(df[target_column])
        logger.debug('Target column encoded')

        # Remove duplicate rows
        df = df.drop_duplicates(keep = 'first')
        logger.debug('Duplicates removed')

        # Apply text transformation to the specified text column
        df.loc[:, text_column] = transform_text(df[text_column])
        logger.debug('Text column transformed')
        return df

    except KeyError as e:
        logger.error('Column not found: %s', e)
        raise
    except Exception as e:
        logger.error('Error during text normalization: %s', e)
        raise


def main(text_column = 'text', target_column = 'target'):
    """
    Main function to load raw data, preprocess it, and save the processed data.
    """       
    try:
        # Fetch the data from data/raw
        train_data = pd.read_csv('./data/raw/train.csv')
        test_data = pd.read_csv('./data/raw/test.csv')
        logger.debug('Data loaded properly')

        # Transform the data
        train_processed_data = preprocess_df(train_data, text_column, target_column)
        test_processed_data = preprocess_df(test_data, text_column, target_column)

        # Store the data inside data/processed
        data_path = os.path.join("./data", "interim")
        os.makedirs(data_path, exist_ok= True)

        train_processed_data.to_csv(os.path.join(data_path, "train_processed.csv"), index = False)
        test_processed_data.to_csv(os.path.join(data_path, "test_processed.csv"), index = False)

        logger.debug('Processed data saved to %s', data_path)

    except FileNotFoundError as e:
        logger.error('File not found: %s', e)
    except pd.errors.EmptyDataError as e:
        logger.error('No data: %s', e)
    except Exception as e:
        logger.error('Failed to complete the data transformation process: %s', e)
        print(f"Error : {e}")


if __name__ == '__main__':
    main()

