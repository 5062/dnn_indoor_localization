import numpy as np
import pandas as pd
import pathlib
from collections import namedtuple
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle


def get_project_root():
    return pathlib.Path(__file__).parent.parent


root = get_project_root()
training_fname = root.joinpath('data/ujiindoorloc/trainingdata.csv')
testing_fname = root.joinpath('data/ujiindoorloc/validationdata.csv')

lack_of_ap = -110


def read_data():
    training_df = pd.read_csv(training_fname).sample(frac=1).reset_index(drop=True)
    testing_df = pd.read_csv(testing_fname).sample(frac=1).reset_index(drop=True)
    col_aps = [col for col in training_df.columns if 'WAP' in col]

    rss_train = np.asarray(training_df[col_aps])
    rss_train[rss_train == 100] = lack_of_ap  # RSS value for lack of AP
    rss_test = np.asarray(testing_df[col_aps])
    rss_test[rss_test == 100] = lack_of_ap  # RSS value for lack of AP
    rss_scaler = StandardScaler()
    rss_scaled_train = rss_scaler.fit_transform(rss_train)
    rss_scaled_test = rss_scaler.transform(rss_test)

    coord_train = np.asarray(training_df[['LONGITUDE', 'LATITUDE']])
    coord_test = np.asarray(testing_df[['LONGITUDE', 'LATITUDE']])
    coord_scaler = StandardScaler()
    coord_scaled_train = coord_scaler.fit_transform(coord_train)
    coord_scaled_test = coord_scaler.transform(coord_test)

    training_df['REFPOINT'] = training_df.apply(
        lambda row: str(int(row['SPACEID'])) + str(int(row['RELATIVEPOSITION'])), axis=1)
    # blds = np.unique(training_df[['BUILDINGID']])  # [0,1,2]
    # flrs = np.unique(training_df[['FLOOR']])  # [0,1,2,3,4]

    training_label_bld = np.asarray(pd.get_dummies(training_df['BUILDINGID']))
    training_label_flr = np.asarray(pd.get_dummies(training_df['FLOOR']))
    training_label_loc = np.asarray(pd.get_dummies(training_df['REFPOINT']))
    testing_label_bld = np.asarray(pd.get_dummies(testing_df['BUILDINGID']))
    testing_label_flr = np.asarray(pd.get_dummies(testing_df['FLOOR']))

    TrainingLabels = namedtuple('TrainingLabels', ['building', 'floor', 'location'])
    TrainingData = namedtuple('TrainingData', [
        'rss', 'rss_scaled', 'rss_scaler', 'coord', 'coord_scaled', 'coord_scaler', 'labels'
    ])
    training_labels = TrainingLabels(
        building=training_label_bld,
        floor=training_label_flr,
        location=training_label_loc
    )
    training_data = TrainingData(
        rss=rss_train,
        rss_scaled=rss_scaled_train,
        rss_scaler=rss_scaler,
        coord=coord_train,
        coord_scaled=coord_scaled_train,
        coord_scaler=coord_scaler,
        labels=training_labels
    )
    TestingLabels = namedtuple('TestingLabels', ['building', 'floor'])
    TestingData = namedtuple('TestingData', ['rss', 'rss_scaled', 'coord', 'coord_scaled', 'labels'])
    testing_labels = TestingLabels(
        building=testing_label_bld,
        floor=testing_label_flr
    )
    testing_data = TestingData(
        rss=rss_test,
        rss_scaled=rss_scaled_test,
        coord=coord_test,
        coord_scaled=coord_scaled_test,
        labels=testing_labels
    )
    return training_data, testing_data


if __name__ == '__main__':
    read_data()
