# Importar bibliotecas necesarias
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Cargar datos de transacciones de transporte
transport_data = pd.read_csv('transport_data.csv')

# Cargar datos de localización
location_data = pd.read_csv('location_data.csv')

# Fusionar datos en función de la fecha o la ubicación
merged_data = pd.merge(transport_data, location_data, on='id_usuario')

# Preprocesamiento de datos
# (puedes agregar más pasos dependiendo de la naturaleza de tus datos)
X = merged_data[['hora', 'estacion_origen', 'latitud', 'longitud']]

# Escalar características (importante para K-means)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Aplicar K-means para el agrupamiento
kmeans = KMeans(n_clusters=3, random_state=42)
merged_data['cluster'] = kmeans.fit_predict(X_scaled)

# Visualizar los resultados (por ejemplo, en un gráfico de dispersión)
plt.scatter(merged_data['latitud'], merged_data['longitud'], c=merged_data['cluster'], cmap='viridis')
plt.title('Agrupamiento de Usuarios')
plt.xlabel('Latitud')
plt.ylabel('Longitud')
plt.show()
