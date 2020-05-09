import pandas as pd
import os
from sklearn import preprocessing
from sklearn import ensemble

TRAINING_DATA = os.environ.get("TRAINING_DATA")
TEST_DATA = os.environ.get("TEST_DATA")
FOLD = os.environ.get("FOLD")
MODEL = os.environ.get("MODEL")

FOLD_MAPPING = {
    0:[1, 2, 3, 4],
    1:[1, 2, 3, 4],
    2:[1, 2, 3, 4],
    3:[1, 2, 3, 4],
    4:[1, 2, 3, 4],
}

if __name__ == "__main__":
    df = pd.read_csv(TRAINING_DATA)
    train_df = df[df.kfold.isin(FOLD_MAPPING.get(FOLD))]
    valid_df = df[df.kfold == FOLD]

    ytrain = train_df.target.values
    yvalid = valid_df.target.values
    train_df = train_df.drop(["id", "target", "kfold"], axis = 1)
    valid_df = valid_df.drop(["id", "target", "kfold"], axis = 1)

    valid_df = valid_df[train_df.columns]

    label_encoders = []
    for c in train_df.columns:
        lbl = preprocessing.LabelEncoder()
        lbl.fit(train_df[c].values.tolist() + valid_df[c].values.tolist())

        train_df.loc[:, c] = lbl.transform(train_df[c]).values.tolist()
        valid_df.loc[:, c] = lbl.transform(valid_df[c]).values.tolist()
        
        label_encoders.append((c,lbl))

    #Ready to train data
    classifier = ensemble.RandomForestClassifier(n_jobs=-1, verbose= 2)
    classifier.fit(train_df, ytrain)
    preds = classifier.predict_proba(valid_df)[:, 1]
    print(preds)