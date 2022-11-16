import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold



iris = load_iris()
# 0 - iris_setosa, 1- iris_versicolor, 2-iris_virginica
# 'sepal length (cm)' ,  'petal length (cm)', 'petal width (cm)', 'sepal width (cm)'
iris_pd = pd.DataFrame(data=np.c_[iris['data'], iris['target']], columns=iris['feature_names'] + ['target'])


def pca(tabl):
    features = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
    tabl = tabl.replace({'target': {0: 'iris_setosa', 1: 'iris_versicolor', 2: 'iris_virginica'}})
    x = tabl.loc[:, features].values
    y = tabl.loc[:, ['target']].values
    x = StandardScaler().fit_transform(x) # pd.DataFrame(data=x, columns=features).head())
    pca = PCA(n_components=2)
    prcp_compon = pca.fit_transform(x)
    prcp_df = pd.DataFrame(data=prcp_compon, columns=['PC1', 'PC2'])
    final_df = pd.concat([prcp_df, tabl['target']], axis=1)
    print(sum(pca.explained_variance_ratio_))
    def pca_graph():
        plt.scatter(final_df['PC1'], final_df['PC2'], label='Хар-ки цветков')
        plt.grid()
        plt.legend()
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title('2 Component PCA')
        plt.show()
    pca_graph()


def tsne(table):
    table = table.drop(columns=['target'])
    features = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
    X = table.loc[:, features].values
    cvetochki = TSNE(n_components=2, perplexity=5, early_exaggeration=12, n_iter=5000, random_state=100)
    X_cvetochki = pd.DataFrame(cvetochki.fit_transform(X))
    plt.scatter(X_cvetochki[0], X_cvetochki[1], label='Хар-ки цветков')
    plt.grid()
    plt.legend()
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title('t-SNE')
    return plt.figure


def lin_reg_class(table):
    kfold = KFold(n_splits=3, shuffle=True)
    max = 0
    for train, test in kfold.split(table):
        lreg_clf = LogisticRegression(max_iter=200)
        lreg_clf.fit(table.iloc[train, [0, 1, 2, 3]], table.iloc[train, [4]])
        lrg_predict = lreg_clf.predict(table.iloc[test, [0, 1, 2, 3]])
        x = [float(accuracy_score(lrg_predict, table.iloc[test, [4]])),
            float(recall_score(lrg_predict, table.iloc[test, [4]], average="macro")),
            float(precision_score(lrg_predict, table.iloc[test, [4]], average="macro")),
            float(f1_score(lrg_predict, table.iloc[test, [4]], average="macro"))]
        if sum(x) > max:
            kfs = x
            max = sum(x)
            test_dats, best_pred_id = table.iloc[test, :], lrg_predict
    [print(f'{k}: {v}') for k, v in dict(zip(['accuracy_score', 'recall_score', 'precision_score', 'f1_score'], map(float, kfs))).items()]
    def graph(x, y, z):
        fig = plt.figure()
        ax = fig.add_subplot(1, 2, 1, projection='3d')
        ax.scatter(test_dats[x], test_dats[y], test_dats[z], c=test_dats.target, label='Хар-ки цвектов')
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_zlabel(z)
        ax.legend()
        plt.title('Реальные классы')
        ax = fig.add_subplot(1, 2, 2, projection='3d')
        ax.scatter(test_dats[x], test_dats[y], test_dats[z], c=best_pred_id,  label='Хар-ки цвектов')
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_zlabel(z)
        ax.legend()
        plt.title('Работа классификатора')
        plt.show()

    graph('sepal length (cm)',  'petal length (cm)', 'petal width (cm)')



