# Importação de bibliotecas
import pandas as pd
import matplotlib.pyplot as plot
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Carregar conjunto de dados
arquivo = "G:/MARCOS VINICIUS/Python/redes MLP E RBF/content/diabetes.csv"
nome_colunas = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]
data = pd.read_csv(arquivo, header = None, names = nome_colunas, skiprows = 1)

# Pré-processar os dados
# Separar features (X) e target (y)
X = data.drop("Outcome", axis = 1)
y = data["Outcome"]

# Dividir os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Normalizar os dados
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Criar e treinar a MLP
mlp = MLPClassifier(hidden_layer_sizes = (10, 10), max_iter = 1000, random_state = 42)  # 2 camadas ocultas com 10 neurônios cada
mlp.fit(X_train, y_train)

# Fazer previsões
y_pred = mlp.predict(X_test)

# Variável de demonstração gráfica
mlp = accuracy_score(y_test, y_pred)

# Avaliar o modelo
print("Acurácia:", accuracy_score(y_test, y_pred))
print("\nMatriz de Confusão:\n", confusion_matrix(y_test, y_pred))
print("\nRelatório de Classificação:\n", classification_report(y_test, y_pred))

# Aplicar a transformação RBF
# Usar RBFSampler para aproximar o kernel RBF
rbf_feature = RBFSampler(gamma = 1, n_components = 100, random_state = 42)
X_train_rbf = rbf_feature.fit_transform(X_train)
X_test_rbf = rbf_feature.transform(X_test)

# Treinar um classificador (usamos Regressão Logística como exemplo)
classifier = LogisticRegression(max_iter = 1000)
classifier.fit(X_train_rbf, y_train)

# Fazer previsões
y_pred = classifier.predict(X_test_rbf)

# Variavel de demonstração gráfica
rbf = accuracy_score(y_test, y_pred)

# Avaliar o modelo
print("Acurácia:", accuracy_score(y_test, y_pred))
print("\nMatriz de Confusão:\n", confusion_matrix(y_test, y_pred))
print("\nRelatório de Classificação:\n", classification_report(y_test, y_pred))

# Dados dos valores de X e Y
categorias = ["MLP", "RBF"]
valores = [mlp, rbf]

# Criar gráfico de barras
plot.bar(categorias, valores)

# Adicionar título e rótulos aos eixos
plot.title("Comparação de Resultados de Acurácia")
plot.xlabel("Modelos de Redes Neurais")
plot.ylabel("Valores Alcançados")

# Exibir o gráfico
plot.show()