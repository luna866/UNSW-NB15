import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, mean_poisson_deviance
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import lightgbm as lgb
import joblib

# Charger les données
train = pd.read_csv(r'E:\CYBERSENTINEL\ML\DF\sample_train10.csv', encoding='ISO-8859-1')

# Définir le mapping des catégories d'attaques
mapping = {'normal':0, 'exploits':1, 'reconnaissance':2, 'dos':3, 'generic':4, 'shellcode':5,'fuzzers':6, 'worms':7, 'backdoor':8, 'analysis':9}

# Imputer les valeurs manquantes pour les variables numériques
numeric_features = train.select_dtypes(include=['float64']).columns
imputer = SimpleImputer(strategy='median')
train[numeric_features] = imputer.fit_transform(train[numeric_features])

# Définir la fonction de transformation Smooth Ridit
def smooth_ridit_transform(data, sparsity_threshold=0.5, skip_bins=True, skip_date_features=True):
    """
    Appliquer la transformation Smooth Ridit aux données numériques.

    Parameters:
    - data: DataFrame
      Données numériques à transformer.
    - sparsity_threshold: float, default=0.5
      Seuil de sparsité. Si la sparsité est supérieure à ce seuil, les données seront centrées sur la médiane.
    - skip_bins: bool, default=True
      Ignorer les caractéristiques binaires.
    - skip_date_features: bool, default=True
      Ignorer les caractéristiques dérivées de la date/heure.

    Returns:
    - DataFrame
      Données transformées.
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
                    ridit_value = np.nan  # ou toute autre valeur qui a du sens dans ton contexte
                ridit_values.append(ridit_value)
            transformed_data[col] = data[col].map(dict(zip(sorted_data, ridit_values)))

    return transformed_data

# Appliquer la transformation Smooth Ridit aux caractéristiques numériques dans le DataFrame d'entraînement
numeric_features = train.select_dtypes(include=[np.number]).columns
train[numeric_features] = smooth_ridit_transform(train[numeric_features])

# Ordinal encoding pour les variables catégorielles
categorical_features = train.select_dtypes(include=['object']).columns

# Vérifier que les colonnes spécifiées sont présentes dans le DataFrame
missing_columns = set(categorical_features) - set(train.columns)
if missing_columns:
    print(f"Colonnes manquantes : {missing_columns}")
else:
    # Appliquer l'encodage ordinal
    ordinal_encoder = OrdinalEncoder()
    train[categorical_features] = ordinal_encoder.fit_transform(train[categorical_features])


# Encoder les variables catégorielles de manière ordinale
ordinal_encoder = OrdinalEncoder()
train[categorical_features] = ordinal_encoder.fit_transform(train[categorical_features])

# Encoder les variables catégorielles de manière One-Hot
encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
encoded_cat = encoder.fit_transform(train[categorical_features])
encoded_df = pd.DataFrame(encoded_cat, columns=encoder.get_feature_names_out(categorical_features))
train.drop(categorical_features, axis=1, inplace=True)  # Supprimer les colonnes catégorielles originales
df_encoded = pd.concat([train, encoded_df], axis=1)

# Régression ElasticNet pour les variables numériques et catégorielles combinées
X = df_encoded.drop('attack_cat', axis=1)
y = df_encoded['attack_cat']


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
imputer = SimpleImputer(strategy='median')
X_train_imputed = imputer.fit_transform(X_train_scaled)
X_val_imputed = imputer.transform(X_val_scaled)
enet = ElasticNet(alpha=0.5, l1_ratio=0.7, max_iter=10000)
enet.fit(X_train_imputed, y_train)
enet_pred_train = enet.predict(X_train_imputed)
enet_pred_val = enet.predict(X_val_imputed)
mse = mean_squared_error(y_val, enet_pred_val)
print(f"Mean Squared Error: {mse}")

# LightGBM avec perte de Poisson
dtrain = lgb.Dataset(X_train, label=y_train)
dval = lgb.Dataset(X_val, label=y_val)
params = {
    'boosting_type': 'gbdt',
    'objective': 'poisson',
    'metric': 'poisson',
    'learning_rate': 0.11,
}

# Définir le callback early_stopping
early_stopping = lgb.callback.early_stopping(stopping_rounds=50, verbose=True)
model = lgb.train(params, dtrain, num_boost_round=1000, valid_sets=[dval], callbacks=[early_stopping])

y_pred = model.predict(X_val, num_iteration=model.best_iteration)
poisson_deviance = mean_poisson_deviance(y_val, y_pred)
print(f"Poisson Deviance: {poisson_deviance}")

# Sauvegarder le modèle et le mapping
joblib.dump(model, 'model.pkl')
joblib.dump(mapping, 'mapping.pkl')

# Charger le modèle et le mapping
model = joblib.load('model.pkl')
mapping = joblib.load('mapping.pkl')