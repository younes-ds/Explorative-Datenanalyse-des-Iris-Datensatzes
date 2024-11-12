# Iris Datensatz Analyse

Dieses Repository enthält ein Python-Projekt zur Analyse des Iris-Datensatzes mit verschiedenen Techniken zur Datenanalyse und -visualisierung. Der Datensatz wird mit Seaborn geladen, und die Zielwerte (Arten) werden aus dem scikit-learn-Datensatz hinzugefügt. Das Projekt führt eine Datenbereinigung, Visualisierung und statistische Analyse durch, um Einblicke in die Beziehungen zwischen den Merkmalen der Iris-Blumen zu gewinnen.

## Datensatz

Der verwendete Datensatz ist der bekannte Iris-Datensatz, der folgende Spalten enthält:

- **sepal_length**: Die Länge des Kelchblattes in Zentimetern.
- **sepal_width**: Die Breite des Kelchblattes in Zentimetern.
- **petal_length**: Die Länge des Blütenblattes in Zentimetern.
- **petal_width**: Die Breite des Blütenblattes in Zentimetern.
- **species**: Die Art der Iris-Blume (setosa, versicolor, virginica).

Der Datensatz wird mit der Funktion `load_dataset('iris')` aus Seaborn geladen, und die Zielwerte werden aus `sklearn.datasets.load_iris()` abgerufen.

## Schritte

1. **Daten Laden**:
   Der Iris-Datensatz wird von Seaborn geladen und mit Zielwerten (Arten) aus scikit-learn erweitert.
   
2. **Datenübersicht**:
   - Die ersten Zeilen des Datensatzes werden angezeigt.
   - Die Form des Datensatzes und die Datentypen der Spalten werden ausgegeben.
   - Fehlende Werte im Datensatz werden überprüft.

3. **Datenbereinigung**:
   Fehlende Werte werden ausgeglichen:
   - Numerische Spalten werden mit dem Mittelwert aufgefüllt.
   - Kategorische Spalten mit dem häufigsten Wert.

4. **Label Encoding**:
   Die `species`-Spalte wird mit `LabelEncoder` aus scikit-learn in numerische Werte umgewandelt.

5. **Datenvisualisierung**:
   - **Balkendiagramm**: Zeigt die Verteilung der Iris-Arten.
   - **Paarplot**: Visualisiert die paarweisen Beziehungen zwischen verschiedenen Merkmalen des Iris-Datensatzes.
   - **Korrelations-Heatmap**: Zeigt die Korrelationen zwischen numerischen Merkmalen.
   - **Boxplot**: Zeigt die Verteilung der Petal-Breite nach Arten.
   - **Balkendiagramm**: Zeigt die Beziehung zwischen Arten und Sepal-Breite.
   - **Liniendiagramm**: Zeigt die Beziehung zwischen Sepal-Länge und Petal-Länge.

6. **Statistische Analyse**:
   Die Korrelationsmatrix zwischen den Merkmalen wird berechnet und visualisiert.

## Visualisierungen

Hier sind einige der erstellten Visualisierungen:

1. **Balkendiagramm**: Verteilung der Iris-Arten.
2. **Paarplot**: Paarweise Beziehungen zwischen Iris-Merkmalen.
3. **Heatmap**: Korrelationsmatrix zwischen numerischen Merkmalen.
4. **Boxplot**: Verteilung der Petal-Breite nach Arten.
5. **Balkendiagramm**: Arten vs. Sepal-Breite.
6. **Liniendiagramm**: Sepal-Länge vs. Petal-Länge.

## Code-Erklärung

### Daten Laden
```python
import seaborn as sns
from sklearn.datasets import load_iris

# Laden des Iris-Datensatzes
df = sns.load_dataset('iris')
iris = load_iris()
df['species'] = iris.target_names[iris.target]

# Datenbereinigung
def clean_data(df):
    for column in df.columns:
        if df[column].dtype == 'object':
            df[column] = df[column].fillna(df[column].mode()[0])
        else:
            df[column] = df[column].fillna(df[column].mean())
    return df

df = clean_data(df)

# Label Encoding
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
df['species'] = label_encoder.fit_transform(df['species'])

#Datenvisualisierungen
import seaborn as sns
import matplotlib.pyplot as plt

# Visualisierungen erstellen
sns.countplot(x='species', data=df, hue='species', palette='Set1')
sns.pairplot(df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']], hue='species', palette='Set2')
sns.heatmap(df.corr(), cmap='coolwarm')
plt.show()

#Autor
Chemingui, Younes
