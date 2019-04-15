import pandas as pd


def add_mrns(data, file_with_mrns):
    mrn_data = pd.read_csv(file_with_mrns)
    mrn_data = mrn_data[['study_id', 'MRN']]
    merged_data = pd.merge(data, mrn_data, on=['study_id'])
    return merged_data

def determine_surgery_laterality(data):
    data['hydro_kidney'] = data['Laterality']
    data.loc[data['Laterality'] == "Bilateral", ['hydro_kidney']] = data["If bilateral, which is the most severe kidney?"]
    return data

def binarize_surgery(data):
    data.loc[data['Surgery'] == "Yes", ['Surgery']] = 1
    data.loc[data['Surgery'] == "No", ['Surgery']] = 0
    return data

def select_columns(data, list_of_columns):
    selected_data = data[list_of_columns]
    if "If bilateral, which is the most severe kidney?" and 'Laterality' in list_of_columns:
        selected_data = determine_surgery_laterality(selected_data)
        selected_data.drop(["If bilateral, which is the most severe kidney?"], axis=1, inplace=True)
    if "Surgery" in list_of_columns:
        selected_data = binarize_surgery(selected_data)
    return selected_data


def load_data(file_with_labels, file_with_mrns, columns_to_extract):
    data = pd.read_csv(file_with_labels)
    data.rename(columns={'Study ID': 'study_id'}, inplace=True)
    data = add_mrns(data, file_with_mrns)
    data = select_columns(data, columns_to_extract)
    return data
