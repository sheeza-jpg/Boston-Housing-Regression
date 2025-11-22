import pandas as pd
import numpy as np 
from sklearn.ensemble import RandomForestRegressor 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
df = pd.read_csv(r"c:\Users\Sheeza\AI Learning\boston.csv")
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())
df['TAX_PER_RM'] = df['TAX'] / df['RM']
df['Nox_indus'] = df['NOX'] * df['INDUS']
df['New_hosue'] = (df['AGE'] < 50).astype(int)
df['CRIM_log'] = np.log1p(df['CRIM'])
df['Tax_log'] = np.log1p(df['TAX'])
df_model = df.drop(['B'], axis=1)
X = df_model.drop('MEDV', axis=1)
y = df_model['MEDV']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42) 
model.fit(X_train, y_train)
pred = model.predict(X_test)
mse = mean_squared_error(y_test, pred)
r2 = r2_score(y_test, pred)
print(f"Mean Squared Error: {mse:.2f}")
print(f"R2 Score: {r2:.2f}")

