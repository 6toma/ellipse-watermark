import pandas as pd
import numpy as np
from math import floor


# Method to permute columns
def permute_columns(df: pd.DataFrame, p, seed):
    # Step 1: Sample p fraction of the columns
    sampled_columns = df.sample(frac=p, axis=1, random_state=seed)  # random_state is used for reproducibility

    # Step 2: Shuffle the sampled columns
    shuffled_columns = sampled_columns.sample(frac=1, axis=1, random_state=seed).reset_index(drop=True)

    # Step 3: Replace the original columns with the shuffled columns
    df[sampled_columns.columns] = shuffled_columns.values

    return df


# Method to permute rows
def permute_rows(df: pd.DataFrame, p, seed):
    sample_df = df.sample(frac=p, random_state=seed)

    # Step 2: Shuffle the sampled rows
    shuffled_sample_df = sample_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Step 3: Replace the original rows with the shuffled rows
    df.loc[sample_df.index] = shuffled_sample_df.values

    return df


# Method to delete columns
def delete_columns(df: pd.DataFrame, p, seed=1):
    cols_to_delete = df.sample(axis=1, frac=p, random_state=seed).columns
    print('deleted cols', cols_to_delete)
    return df.drop(cols_to_delete, axis=1)


# Method to delete rows
def delete_rows(df: pd.DataFrame, p=1, seed=1):
    return df.drop(df.sample(frac=p, random_state=seed).index)


def duplicate_rows(df, num_duplicates=1):
    df_copy = df.copy()
    rows_to_duplicate = df.sample(n=num_duplicates, random_state=1)
    df_copy = pd.concat([df_copy, rows_to_duplicate])
    return df_copy.reset_index(drop=True)


# Method to add random noise
def add_random_noise_num(df: pd.DataFrame, noise_level=0.1):
    noisy_df = df.copy()
    for col in noisy_df.select_dtypes(include=np.number).columns:
        noise = np.random.normal(0, noise_level, size=noisy_df[col].shape)
        noisy_df[col] += noise
    return noisy_df


def add_random_noise_cat(df, noise_level=0.1):
    # Copy the DataFrame to avoid modifying the original one
    df_noisy = df.copy()

    # Identify categorical columns
    categorical_cols = df_noisy.select_dtypes(include=['object', 'category']).columns

    for col in categorical_cols:
        # Calculate the number of values to replace
        num_values = len(df_noisy[col])
        num_to_replace = int(noise_level / 100 * num_values)

        if num_to_replace == 0:
            continue

        # Randomly select indices to replace
        indices_to_replace = np.random.choice(df_noisy.index, size=num_to_replace, replace=False)

        # Randomly select replacement values from the same column
        replacement_values = np.random.choice(df_noisy[col], size=num_to_replace, replace=True)

        # Replace the selected values with the replacement values
        df_noisy.loc[indices_to_replace, col] = replacement_values

    return df_noisy


# Method to insert outliers
def insert_outliers(df, columns, magnitude=10, num_outliers=1):
    df_copy = df.copy()
    for col in columns:
        outlier_indices = np.random.choice(df_copy.index, num_outliers, replace=False)
        df_copy.loc[outlier_indices, col] *= magnitude
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


# Complex attack sequence
def attack_sequence(df, strength=0):
    if strength == 0:
        return df
    elif strength == 1:
        attack_occurrence = 0.1
    elif strength == 2:
        attack_occurrence = 0.25
    elif strength == 3:
        attack_occurrence = 0.5
    elif strength == 4:
        attack_occurrence = 0.75
    else:
        attack_occurrence = 1
        print("Too strong!")

    total_columns = df.shape[1]
    total_rows = df.shape[0]

    num_to_select = max(1, int(total_columns * attack_occurrence))
    random_columns = np.random.choice(total_columns, size=num_to_select, replace=False)
    columns = df.columns[random_columns]

    rows = floor(total_rows * attack_occurrence)

    df = add_random_noise_num(df, attack_occurrence)
    print(f' -- added {100 * attack_occurrence}% random noise to numerical columns')

    df = insert_outliers(df, columns, 10, rows)
    print(f' -- inserted {rows} outliers to columns {columns}')

    df = scale_values(df, columns, 2)
    print(f' -- scaled values by 300% on columns {columns}')

    df = duplicate_rows(df, rows)
    print(f' -- duplicated {rows} rows')

    df = add_random_noise_cat(df, attack_occurrence)
    print(f' -- added {100 * attack_occurrence}% random noise to categorical columns')

    rows = rows // 2
    df = delete_rows(df, rows)
    print(f' -- deleted {rows} rows')

    if strength >= 3:
        df = permute_columns(df)
        print(' -- permuted all columns randomly')

        df = permute_rows(df)
        print(' -- permuted all rows randomly')

    return df
