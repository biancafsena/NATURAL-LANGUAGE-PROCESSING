# -*- coding: utf-8 -*-
"""Projeto Integrado.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1MWCC5Y-rPscD0w69FTfxggIFgFyDe7OX

# **Parte 1 – Criar um modelo classificador de assuntos aplicando técnicas tradicionais de NLP, que consiga classificar através de um texto o assunto conforme disponível na base de dados[1] para treinamento e validação do modelo seu modelo.**
"""

!pip install openai

!pip install openai --upgrade
!openai migrate

pip install pandas scikit-learn openai

pip install scikit-plot

"""### **Bibliotecas**"""

import pandas as pd
import openai
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import scikitplot as skplt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

"""Com objetivo de importar varias bibliotecas com modulos de dados.

### **Carregando dados**
"""

url_tradicional = "https://raw.githubusercontent.com/thiagonogueira/datasets/main/tickets_reclamacoes_classificados_one_line.csv"
df_tradicional = pd.read_csv(url_tradicional, delimiter=';')

"""Possui objetivo de carregar dados de um arquivo CSV hospedado na internet para um DataFrame usando a biblioteca pandas em Python.

### **Verificação de nomes das colunas**
"""

print("Nomes das Colunas:", df_tradicional.columns)

"""Possui a função de imprimir os nomes das colunas do DataFrame df_tradicional.

### **Definição de coluna de texto**
"""

text_column_tradicional = df_tradicional.columns[3]

"""Com o objetivo de atribuir o nome da terceira coluna do DataFrame df_tradicional à variável text_column_tradicional.

### **Divisão de dados em treino e teste**
"""

train_data_tradicional, test_data_tradicional, train_labels_tradicional, test_labels_tradicional = train_test_split(
    df_tradicional[text_column_tradicional],
    df_tradicional['categoria'],
    test_size=0.2,
    random_state=42
)

"""Para dividir os dados do DataFrame df_tradicional em conjuntos de treinamento e teste, incluindo os textos das reclamações (text_column_tradicional) e as categorias (categoria).

### **Criação de vetorizador TF-IDF**
"""

tfidf_vectorizer_tradicional = TfidfVectorizer(stop_words=None)
train_vectors_tradicional = tfidf_vectorizer_tradicional.fit_transform(train_data_tradicional)
test_vectors_tradicional = tfidf_vectorizer_tradicional.transform(test_data_tradicional)

"""Realizar a vetorização dos textos de treinamento e teste usando a técnica TF-IDF.

## **Treinamento do modelo**
"""

clf_tradicional = MultinomialNB()
clf_tradicional.fit(train_vectors_tradicional, train_labels_tradicional)

"""Possui a função de criar e treinar um classificador Naive Bayes Multinomial usando os vetores de treinamento gerados pela técnica TF-IDF.

### **Avaliação do modelo**
"""

predictions_tradicional = clf_tradicional.predict(test_vectors_tradicional)
print("Acurácia Tradicional:", accuracy_score(test_labels_tradicional, predictions_tradicional))
print("\nRelatório de Classificação Tradicional:\n", classification_report(test_labels_tradicional, predictions_tradicional))

"""Realizar previsões usando o modelo treinado clf_tradicional nos dados de teste e, em seguida, imprimir métricas de avaliação do desempenho do modelo.

### **Calculando a Matriz de Confusão**
"""

conf_matrix = confusion_matrix(test_labels_tradicional, predictions_tradicional)

"""Com a função de calcular a matriz de confusão com base nas previsões do modelo clf_tradicional nos dados de teste.

### **Visualização - Matriz de Confusão**
"""

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=clf_tradicional.classes_, yticklabels=clf_tradicional.classes_)
plt.xlabel('Previsão')
plt.ylabel('Real')
plt.title('Matriz de Confusão')
plt.show()

"""Possui o objetico de criar e exibir um mapa de calor (heatmap) visualizando a matriz de confusão calculada anteriormente.

### **Visualização - Curva ROC**
"""

y_probas = clf_tradicional.predict_proba(test_vectors_tradicional)
skplt.metrics.plot_roc(test_labels_tradicional, y_probas, figsize=(8, 6), plot_micro=False)
plt.title('Curva ROC Tradicional')
plt.show()

"""Com o objetivo de plotar a Curva ROC (Receiver Operating Characteristic) para avaliar o desempenho do modelo de classificação Naive Bayes Multinomial tradicional nos dados de teste.

# **Parte 2 – Realizar a tarefa de classificação apresentada no item anterior com a utilização IA Generativa.**
"""



"""# **Parte 3 (Extra) – Utilizar a IA Generativa para fazer uma classificação livre de assuntos e avaliar qualitativamente os resultados.**"""