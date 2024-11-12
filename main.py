import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.preprocessing import LabelEncoder

# Lade den Iris-Datensatz von Seaborn
df = sns.load_dataset('iris')

# Lade den Iris-Datensatz aus sklearn (für die Zielwerte)
iris = load_iris()

# Füge die Zielspalte ('species') aus dem sklearn-Datensatz hinzu
df['species'] = iris.target_names[iris.target]

# Überblick über den Datensatz verschaffen
print(df.head())  # Zeige die ersten Zeilen des Datensatzes
print(f'Dimensionen des Datensatzes: {df.shape}')  # Form des Datensatzes anzeigen
print(f'Datentypen der Spalten:\n{df.dtypes}')  # Datentypen anzeigen
print(f'Fehlende Werte in den Spalten:\n{df.isnull().sum()}')  # Überprüfe auf fehlende Werte

# Funktion zur Bereinigung von fehlenden Werten
def clean_data(df):
    """
    Fehlende Werte im Datensatz auffüllen: 
    Für numerische Spalten wird der Mittelwert verwendet, 
    für kategorische Spalten der häufigste Wert (Mode).
    """
    for column in df.columns:
        if df[column].dtype == 'object':  # Für kategorische Spalten
            df[column] = df[column].fillna(df[column].mode()[0])  # Mode verwenden
        else:  # Für numerische Spalten
            df[column] = df[column].fillna(df[column].mean())  # Mittelwert verwenden
    return df

# Bereinigung anwenden
df = clean_data(df)

# Umwandlung der 'species'-Spalte in numerische Werte mit Label-Encoding
label_encoder = LabelEncoder()
df['species'] = label_encoder.fit_transform(df['species'])

# Zeige die aktualisierten Informationen an
print(f'Aktualisierte Datentypen:\n{df.dtypes}')
print(f'Erste 5 Zeilen des Datensatzes:\n{df.head()}')
print(f'Beschreibende Statistiken:\n{df.describe()}')  # Statistische Kennzahlen anzeigen
print(f'Verteilung der Arten:\n{df["species"].value_counts()}')  # Häufigkeit der Arten anzeigen

# Setze die Farbpalette für die Arten
palette = sns.color_palette("Set2", n_colors=3)

# Erstelle ein Balkendiagramm zur Verteilung der Arten
plt.figure(figsize=(8, 6))
sns.countplot(x='species', data=df, hue='species', palette='Set1')
plt.title('Verteilung der Iris-Arten')
plt.xlabel('Arten')
plt.ylabel('Anzahl')
plt.show()

# Berechne die Korrelation zwischen 'sepal_length' und 'petal_length'
correlation = df[['sepal_length', 'petal_length']].corr()
print(f'Korrelationsmatrix zwischen Sepal-Länge und Petal-Länge:\n{correlation}')

# Erstelle einen Paarplot, um Beziehungen zwischen verschiedenen Variablen zu visualisieren
sns.pairplot(df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']], hue='species', palette=palette)
plt.suptitle('Paarweise Beziehungen zwischen Iris-Merkmalen')
plt.show()

# Wähle nur die numerischen Spalten aus
numeric_df = df.select_dtypes(include=[np.number])

# Erstelle eine Heatmap der Korrelationen
sns.heatmap(numeric_df.corr(), cmap='coolwarm')
plt.title('Korrelations-Heatmap')
plt.show()

# Erstelle ein Boxplot für die Petal-Breite nach Arten
sns.boxplot(x='species', y='petal_width', data=df, hue='species', palette='Set1')
plt.title('Petal-Breite nach Arten')
plt.xlabel('Arten')
plt.ylabel('Petal-Breite (cm)')
plt.show()

# Erstelle ein Balkendiagramm für Arten vs. Sepal-Breite
sns.barplot(x='species', y='sepal_width', data=df, hue='species', palette='Set1')
plt.title('Arten vs. Sepal-Breite')
plt.xlabel('Arten')
plt.ylabel('Sepal-Breite (cm)')
plt.show()

# Erstelle ein Liniendiagramm für Sepal-Länge vs. Petal-Länge, gefärbt nach Arten
sns.lineplot(x='sepal_length', y='petal_length', data=df, hue='species', palette=palette)
plt.title('Sepal-Länge vs. Petal-Länge')
plt.xlabel('Sepal-Länge (cm)')
plt.ylabel('Petal-Länge (cm)')
plt.show()

# Berechne die Korrelationsmatrix für alle numerischen Spalten
corr_matrix = numeric_df.corr()

# Erstelle eine Heatmap der Korrelationsmatrix
sns.heatmap(corr_matrix, cmap='coolwarm')
plt.title('Korrelationsmatrix-Heatmap')
plt.show()
