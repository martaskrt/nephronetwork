import pandas as pd

obstr_etiology_labels = "PHN-ObstructiveEtiologie_Labels.csv"
reflux_etiology_labels = 'PHN-RefluxEtiologiesML_Labels.csv'


class RenameDupCols():
    def __init__(self):
        self.d = dict()

    def __call__(self, x):
        if x not in self.d:
            self.d[x] = 0
            return x
        else:
            self.d[x] += 1
            return "%s_%d" % (x, self.d[x])

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

def group_etiology(data):
    obstr_data = pd.read_csv(obstr_etiology_labels)
    obstr_data.rename(columns=RenameDupCols())
    obstr_data = obstr_data[['study_id', 'etiology', 'Surgery', 'Type']]
    obstr_data['etiology'] = 1

    reflux_data = pd.read_csv(reflux_etiology_labels)
    reflux_data.rename(columns=RenameDupCols())
    reflux_data = reflux_data[['study_id', 'etiology', 'Surgery', 'Type']]
    reflux_data['etiology'] = 0

    merged_data = pd.merge(data, obstr_data, on=['study_id'])
    merged_data = pd.merge(merged_data, reflux_data, on=['study_id'])

    merged_data.loc[merged_data['Surgery'] == "Yes" and merged_data['Type'] != "Circumcision", ['Surgery']] = 1
    merged_data.loc[merged_data['Surgery'] == "Yes" and merged_data['Type'] == "Circumcision", ['Surgery']] = 0
    merged_data.loc[merged_data['Surgery'] == "No", ['Surgery']] = 0

    return merged_data

def select_columns(data, list_of_columns, etiology):
    selected_data = data[list_of_columns]
    if "If bilateral, which is the most severe kidney?" and 'Laterality' in list_of_columns:
        selected_data = determine_surgery_laterality(selected_data)
        selected_data.drop(["If bilateral, which is the most severe kidney?"], axis=1, inplace=True)
    if etiology:
        selected_data = group_etiology(selected_data)
    elif "Surgery" in list_of_columns:
        selected_data = binarize_surgery(selected_data)
    return selected_data


def load_data(file_with_labels, file_with_mrns, columns_to_extract, etiology=True):
    data = pd.read_csv(file_with_labels)
    data.rename(columns={'Study ID': 'study_id'}, inplace=True)
    data = add_mrns(data, file_with_mrns)
    data = select_columns(data, columns_to_extract, etiology)
    return data
