import numpy as np
import pandas as pd


from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

import lightgbm as lgb

import joblib

train = pd.read_csv(r'E:\M2i\RNCP\RNCP_CDC\DataBase\sample_train10.csv', encoding='ISO-8859-1')

mapping = {'normal':0, 'exploits':1, 'reconnaissance':2, 'dos':3, 'generic':4, 'shellcode':5,'fuzzers':6, 'worms':7, 'backdoor':8, 'analysis':9}

# Étape 1 : Impute Missing Values for Numeric Variables
from sklearn.impute import SimpleImputer

# Assuming df is your DataFrame
numeric_features = train.select_dtypes(include=['float64']).columns
imputer = SimpleImputer(strategy='median')
train[numeric_features] = imputer.fit_transform(train[numeric_features])

# Étape 2 : Smooth Ridit Transform for Numeric Variables
def smooth_ridit_transform(data, sparsity_threshold=0.5, skip_bins=True, skip_date_features=True):
    """
    Apply Smooth Ridit Transform to numeric data.

    Parameters:
    - data: DataFrame
      Numeric data to be transformed.
    - sparsity_threshold: float, default=0.5
      Threshold for sparsity. If the sparsity is higher than this threshold, data will be centered to the median.
    - skip_bins: bool, default=True
      Whether to skip binary features.
    - skip_date_features: bool, default=True
      Whether to skip date/time derived features.

    Returns:
    - DataFrame
      Transformed data.
    """
    transformed_data = data.copy()
    for col in data.columns:
        if skip_bins and np.isin(data[col].dropna().unique(), [0, 1]).all():
            continue
        if skip_date_features and np.issubdtype(data[col].dtype, np.datetime64):
            continue
        sparsity = (data[col] == 0).mean()
        if sparsity > sparsity_threshold:
            transformed_data[col] = data[col] - data[col].median()
        else:
            sorted_data = data[col].sort_values()
            n = len(sorted_data)
            ridit_values = []
            for i in range(n):
                if sorted_data.iloc[i] != 0:
                    ridit_value = (sorted_data.iloc[:i+1].sum() + sorted_data.iloc[i+1:].sum()) / (2 * n * sorted_data.iloc[i])
                else:
                    ridit_value = np.nan  # or any other value that makes sense in your context
                ridit_values.append(ridit_value)
            transformed_data[col] = data[col].map(dict(zip(sorted_data, ridit_values)))

    return transformed_data

# Apply Smooth Ridit Transform to numeric features in the train DataFrame
numeric_features = train.select_dtypes(include=[np.number]).columns
train[numeric_features] = smooth_ridit_transform(train[numeric_features])

# Étape 3 : Encodage One-Hot pour les variables catégorielles

categorical_features = train.select_dtypes(include=['object']).columns
encoder = OneHotEncoder(sparse_output=False, drop='first')
encoded_cat = encoder.fit_transform(train[categorical_features])

# Création d'un DataFrame à partir de notre tableau numpy
encoded_df = pd.DataFrame(encoded_cat, columns=encoder.get_feature_names_out(categorical_features))

# Suppression des colonnes catégorielles originales du df original
train.drop(categorical_features, axis=1, inplace=True)

# Concaténation du dataframe original avec le dataframe encodé
df_encoded = pd.concat([train, encoded_df], axis=1)

# Étape 4 : Régression Elastic-Net pour les variables numériques et catégorielles combinées
# On suppose que X contient vos caractéristiques et y est votre variable cible

# 'attack_cat' est votre variable cible
X = df_encoded.drop('attack_cat', axis=1)
y = df_encoded['attack_cat']

# Diviser les données en ensembles d'entraînement et de test
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Maintenant, vous pouvez continuer avec la mise à l'échelle et l'imputation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

imputer = SimpleImputer(strategy='median')
X_train_imputed = imputer.fit_transform(X_train_scaled)
X_val_imputed = imputer.transform(X_val_scaled)

# Entraîner le modèle ElasticNet
enet = ElasticNet(alpha=0.5, l1_ratio=0.7, max_iter=10000)
enet.fit(X_train_imputed, y_train)

# Faire des prédictions
y_pred = enet.predict(X_val_imputed)

# Calculer l'erreur quadratique moyenne
mse = mean_squared_error(y_val, y_pred)
print(f"Erreur quadratique moyenne : {mse}")

# Sauvegarder le modèle
model = enet

# Étape 5 : Codage ordinal pour les variables catégorielles
from sklearn.preprocessing import OrdinalEncoder
import joblib

def YOUR_ORDINAL_ENCODING_FUNCTION(data):
    # Création de l'encodeur ordinal
    ordinal_encoder = OrdinalEncoder()
    
    # Entraînement de l'encodeur et transformation des données
    data_encoded = ordinal_encoder.fit_transform(data)
    
    # Création d'un DataFrame à partir des données encodées
    df_encoded = pd.DataFrame(data_encoded, columns=data.columns)
    
    return df_encoded

# Appliquer le codage ordinal aux caractéristiques catégorielles dans le DataFrame d'entraînement
categorical_features = train.select_dtypes(include=['object']).columns
df_categorical = YOUR_ORDINAL_ENCODING_FUNCTION(train[categorical_features])

#Step 6: Prédictions sur l'ensemble de données en utilisant le modèle LightGBM (Light Gradient Boosting) entraîné.

# 'attack_cat' est votre variable cible
X = df_encoded.drop('attack_cat', axis=1)
y = df_encoded['attack_cat']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.0824, random_state=42)

params = {
    'boosting_type': 'gbdt',
    'objective': 'poisson',
    'metric': 'poisson',
    'learning_rate': 0.11,
}

dtrain = lgb.Dataset(X_train, label=y_train)
dval = lgb.Dataset(X_val, label=y_val)

# Définir le callback early_stopping
early_stopping = lgb.callback.early_stopping(stopping_rounds=50, verbose=True)

model = lgb.train(params, dtrain, num_boost_round=1000, valid_sets=[dval], callbacks=[early_stopping])

# Faire des prédictions sur l'ensemble de données complet
y_pred = model.predict(X, num_iteration=model.best_iteration)

# Round the predictions and convert to integers
y_pred_rounded = np.round(y_pred).astype(int)

# Create an inverse mapping dictionary
inverse_mapping = {v: k for k, v in mapping.items()}

# Convert the rounded predictions to categories
y_pred_categories = [inverse_mapping[i] for i in y_pred_rounded]

# Now y_pred_categories contains the predicted categories


# Save the model and the mapping
joblib.dump(model, 'model.joblib')
joblib.dump(mapping, 'mapping.joblib')

# Load the model
model = joblib.load('model.joblib')

# Load the mapping dictionary
mapping = joblib.load('mapping.joblib')

new_data=pd.read_csv(r'E:\M2i\RNCP\RNCP_CDC\DataBase\sample_train5.csv', encoding='ISO-8859-1')

def preprocess_data(data, imputer, encoder, scaler, train_columns):
    # Ajouter des fonctionnalités manquantes
    missing_cols = set(train_columns) - set(data.columns)
    for c in missing_cols:
        data[c] = np.nan

    # Assurez-vous que l'ordre des colonnes est le même que dans les données d'entraînement
    data = data[train_columns]

    # Impute missing values
    numeric_features = data.select_dtypes(include=['float64']).columns
    data[numeric_features] = imputer.transform(data[numeric_features])
    
    # Apply Smooth Ridit Transform
    numeric_features = data.select_dtypes(include=[np.number]).columns
    data[numeric_features] = smooth_ridit_transform(data[numeric_features])
    
    # One-Hot encoding
    categorical_features = data.select_dtypes(include=['object']).columns
    encoded_cat = encoder.transform(data[categorical_features])
    encoded_df = pd.DataFrame(encoded_cat, columns=encoder.get_feature_names_out(categorical_features))
    
    # Ajouter des colonnes manquantes après l'encodage One-Hot
    missing_cols_encoded = set(train_columns) - set(encoded_df.columns)
    for c in missing_cols_encoded:
        encoded_df[c] = 0

    # Assurez-vous que l'ordre des colonnes est le même que dans les données d'entraînement
    encoded_df = encoded_df[train_columns]
    
    # Scale and impute
    data_scaled = scaler.transform(encoded_df)
    data_imputed = imputer.transform(data_scaled)
    
    return data_imputed

# Suppose new_data is your new DataFrame
new_data_preprocessed = preprocess_data(new_data, imputer, encoder, scaler, X_train.columns)

# Use the model to make predictions
predictions = model.predict(new_data_preprocessed)
