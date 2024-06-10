import pandas as pd
import numpy as np


# Method to permute columns
def permute_columns(df: pd.DataFrame):
    return df.sample(frac=1, axis=1).reset_index(drop=True)


# Method to permute rows
def permute_rows(df: pd.DataFrame):
    return df.sample(frac=1).reset_index(drop=True)


# Method to delete columns
def delete_columns(df: pd.DataFrame, num_columns=1):
    return df.drop(df.sample(axis=1, n=num_columns, random_state=1).columns, axis=1)


# Method to delete rows
def delete_rows(df: pd.DataFrame, num_rows=1):
    return df.drop(df.sample(n=num_rows, random_state=1).index)


# Method to add random noise
def add_random_noise(df: pd.DataFrame, noise_level=0.01):
    noisy_df = df.copy()
    for col in noisy_df.select_dtypes(include=np.number).columns:
        noise = np.random.normal(0, noise_level, size=noisy_df[col].shape)
        noisy_df[col] += noise
    return noisy_df


def duplicate_rows(df, num_duplicates=1):
    df_copy = df.copy()
    rows_to_duplicate = df.sample(n=num_duplicates, random_state=1)
    df_copy = pd.concat([df_copy, rows_to_duplicate])
    return df_copy.reset_index(drop=True)


# Method to change data types
def change_data_types(df, columns, new_type):
    df_copy = df.copy()
    for col in columns:
        df_copy[col] = df_copy[col].astype(new_type)
    return df_copy


# Method to insert outliers
def insert_outliers(df, columns, magnitude=10, num_outliers=1):
    df_copy = df.copy()
    for col in columns:
        outlier_indices = np.random.choice(df_copy.index, num_outliers, replace=False)
        df_copy.loc[outlier_indices, col] *= magnitude
    return df_copy


# Method to replace values
def replace_values(df, column, old_value, new_value):
    df_copy = df.copy()
    df_copy[column].replace(old_value, new_value, inplace=True)
    return df_copy


# Method to scale values
def scale_values(df, columns, scale_factor=1.0):
    df_copy = df.copy()
    for col in columns:
        df_copy[col] *= scale_factor
    return df_copy


def shift_values(df, columns, shift_amount=1.0):
    df_copy = df.copy()
    for col in columns:
        df_copy[col] += shift_amount
    return df_copy


# Method to swap values between two columns
def swap_values(df, column1, column2):
    df_copy = df.copy()
    df_copy[[column1, column2]] = df_copy[[column1, column2]].values[:, ::-1]
    return df_copy


# Method to corrupt data
def corrupt_data(df, corruption_probability=0.1):
    df_copy = df.copy()
    for col in df_copy.columns:
        mask = np.random.rand(len(df_copy)) < corruption_probability
        df_copy.loc[mask, col] = np.nan
    return df_copy


# Method to augment data (simple row duplication)
def augment_data(df, augmentation_factor=0.1):
    df_copy = df.copy()
    num_rows_to_add = int(len(df) * augmentation_factor)
    rows_to_duplicate = df.sample(n=num_rows_to_add, random_state=1)
    df_copy = pd.concat([df_copy, rows_to_duplicate])
    return df_copy.reset_index(drop=True)


# Method to randomize specific columns
def randomize_columns(df, columns):
    df_copy = df.copy()
    for col in columns:
        df_copy[col] = np.random.permutation(df_copy[col].values)
    return df_copy


# Simple attack sequence
def simple_attack_sequence(df):
    df = permute_rows(df)
    df = add_random_noise(df, noise_level=0.05)
    return df


# Complex attack sequence
def complex_attack_sequence(df):
    df = delete_columns(df, num_columns=2)
    df = shift_values(df, columns=df.select_dtypes(include=np.number).columns, shift_amount=5)
    df = corrupt_data(df, corruption_probability=0.1)
    return df


# Custom attack sequence
def custom_attack_sequence(df, sequence):
    for attack in sequence:
        if attack['type'] == 'permute_columns':
            df = permute_columns(df)
        elif attack['type'] == 'permute_rows':
            df = permute_rows(df)
        elif attack['type'] == 'delete_columns':
            df = delete_columns(df, num_columns=attack.get('num_columns', 1))
        elif attack['type'] == 'delete_rows':
            df = delete_rows(df, num_rows=attack.get('num_rows', 1))
        elif attack['type'] == 'add_random_noise':
            df = add_random_noise(df, noise_level=attack.get('noise_level', 0.01))
        elif attack['type'] == 'shift_values':
            df = shift_values(df, columns=attack.get('columns', df.select_dtypes(include=np.number).columns), shift_amount=attack.get('shift_amount', 1.0))
        elif attack['type'] == 'swap_values':
            df = swap_values(df, column1=attack['column1'], column2=attack['column2'])
        elif attack['type'] == 'corrupt_data':
            df = corrupt_data(df, corruption_probability=attack.get('corruption_probability', 0.1))
        elif attack['type'] == 'augment_data':
            df = augment_data(df, augmentation_factor=attack.get('augmentation_factor', 0.1))
        elif attack['type'] == 'randomize_columns':
            df = randomize_columns(df, columns=attack['columns'])
    return df
