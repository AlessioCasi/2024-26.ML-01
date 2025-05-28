from sklearn.ensemble import RandomForestClassifier  
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split, KFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import sklearn
import time

# Impostazioni
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
sklearn.set_config(transform_output="pandas")

# Caricamento e preprocessing
df = pd.read_csv("casiroli/output_bilanciato.csv")#carica dataset
df = df[df["Winner"].isin(["Player 1", "Player 2"])]#togli i pareggi

df['P1_Rank'], df['P2_Rank'] = zip(*df['Rank'].str.split('-').tolist())#i rank diventano due colonne

rank_order = {'platinum':2, 'diamond':1, 'master':3}#i rank iniziano ad avere un numero
df['P1_Rank_Value'] = df['P1_Rank'].map(rank_order)
df['P2_Rank_Value'] = df['P2_Rank'].map(rank_order)

df['Rank_Difference'] = df['P1_Rank_Value'] - df['P2_Rank_Value']#si crea un altra colonna che indica la differenza tra rank
#(se è presente molto probabilmente vincera giocatore 1)

# Train/test split
X = df[["Rank_Difference","P1_Rank_Value","P2_Rank_Value", "Player 1", "Player 2", "Stage"]]
y = df["Winner"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=int(time.time()),
    stratify=y
)

#encoder onehot
encoding = ColumnTransformer([
    ("onehot", OneHotEncoder(sparse_output=False, min_frequency=100, handle_unknown="infrequent_if_exist"),
     ["Player 1", "Player 2", "Stage"])
], remainder="passthrough")

#pipeline usando la random forest
pipe = Pipeline([
    ('encoder', encoding),
    ('classifier', RandomForestClassifier(random_state=42))
])

#parametri
params = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__criterion': ['gini', 'entropy'],
    'encoder__onehot__min_frequency': [20, 30, 40]
}

#riunisci tutto quando nella grid search cosi trova il modo piu ideale di approcciare il dataset
grid_search = GridSearchCV(
    estimator=pipe, #encoder e classifier
    param_grid=params, #parametri tipo N alberi e profondita
    n_jobs=-1, #core della cpu da usare
    cv=KFold(n_splits=5, shuffle=True, random_state=42),
    refit=True,
    verbose=4
)

# Fit del modello
grid_search.fit(X_train, y_train)

# Risultati
print("Migliori parametri:", grid_search.best_params_)
print("Miglior score CV:", grid_search.best_score_)

# Predizione
y_pred = grid_search.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Feature importance
importances = grid_search.best_estimator_.named_steps['classifier'].feature_importances_
feature_names = grid_search.best_estimator_.named_steps['encoder'].get_feature_names_out()
feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
print(feat_imp.head(10))
