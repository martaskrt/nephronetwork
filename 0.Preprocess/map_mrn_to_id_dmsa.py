import pandas as pd
DMSA_sheet = "DMSA_Mandy.csv"
mrn2studyid = "samples_with_studyids_and_mrns.csv"


def main():
    dmsa_data = pd.read_csv(DMSA_sheet)
    dmsa_data.columns = dmsa_data.columns.str.strip().str.lower().str.replace(" ", "_")
    # dmsa_data["mrn"].apply(float)
    # dmsa_data["mrn"].apply(str)
    print(dmsa_data)


    mrn_data = pd.read_csv(mrn2studyid)
    mrn_data.columns = mrn_data.columns.str.strip().str.lower().str.replace(" ", "_")

    mrn_data["mrn"].apply(int)
    mrn_data["mrn"].apply(str)
    mrn_data = mrn_data[['study_id', 'mrn']]
    print(mrn_data)
    merged_data = pd.merge(dmsa_data, mrn_data, on=['mrn'])
    data = merged_data
    # data = merged_data.drop(["mrn"], axis=1)
    # data = data.drop(["accession_#"], axis=1)
    print(data)
    print(data.columns)
    data.to_csv("dmsa_with_studyids_20190604.csv", sep=',')

if __name__ == "__main__":
    main()