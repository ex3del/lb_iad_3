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
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


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


def k_sosedi(table):
    kfold = KFold(n_splits=3, shuffle=True, random_state=110)
    max = 0
    for train, test in kfold.split(table):
        knn_mod = KNeighborsClassifier(n_neighbors=3)
        knn_mod.fit(table.iloc[train, [0, 1, 2, 3]], table.iloc[train, [4]])
        knn_pred = knn_mod.predict(table.iloc[test, [0, 1, 2, 3]])
        ks = [float(accuracy_score(knn_pred, table.iloc[test, [4]])),
            float(recall_score(knn_pred, table.iloc[test, [4]], average="macro")),
            float(precision_score(knn_pred, table.iloc[test, [4]], average="macro")),
            float(f1_score(knn_pred, table.iloc[test, [4]], average="macro"))]
        if sum(ks) > max and sum(ks) != 1:
            kfs = ks
            max = sum(ks)
            test_dats, best_pred_id = table.iloc[test, :], knn_pred
    [print(f'{k}: {v}') for k, v in dict(zip(['accuracy_score', 'recall_score', 'precision_score', 'f1_score'], map(float, kfs))).items()]
    print(f'average: {sum(ks)/len(ks)}')

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


def lin_reg_class(table):
    kfold = KFold(n_splits=3, shuffle=True, random_state=110)
    max = 0
    for train, test in kfold.split(table):
        lreg_clf = LogisticRegression(max_iter=100)
        lreg_clf.fit(table.iloc[train, [0, 1, 2, 3]], table.iloc[train, [4]])
        lrg_predict = lreg_clf.predict(table.iloc[test, [0, 1, 2, 3]])
        ks = [float(accuracy_score(lrg_predict, table.iloc[test, [4]])),
             float(recall_score(lrg_predict, table.iloc[test, [4]], average="macro")),
             float(precision_score(lrg_predict, table.iloc[test, [4]], average="macro")),
             float(f1_score(lrg_predict, table.iloc[test, [4]], average="macro"))]
        if sum(ks) > max and sum(ks) != 1:
            kfs = ks
            max = sum(ks)
            test_dats, best_pred_id = table.iloc[test, :], lrg_predict
    [print(f'{k}: {v}') for k, v in dict(zip(['accuracy_score', 'recall_score', 'precision_score', 'f1_score'], map(float, kfs))).items()]
    print(f'average: {sum(ks)/len(ks)}')

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
        ax.scatter(test_dats[x], test_dats[y], test_dats[z], c=best_pred_id, label='Хар-ки цвектов')
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_zlabel(z)
        ax.legend()
        plt.title('Работа классификатора')
        plt.show()

    graph('sepal length (cm)', 'petal length (cm)', 'petal width (cm)')


def drevo(table):
    kfold = KFold(n_splits=3, shuffle=True, random_state=110)
    max = 0
    for train, test in kfold.split(table):
        drevo_clf = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=110)
        drevo_clf.fit(table.iloc[train, [0, 1, 2, 3]], table.iloc[train, [4]])
        drevo_predict = drevo_clf.predict(table.iloc[test, [0, 1, 2, 3]])
        ks = [float(accuracy_score(drevo_predict, table.iloc[test, [4]])),
             float(recall_score(drevo_predict, table.iloc[test, [4]], average="macro")),
             float(precision_score(drevo_predict, table.iloc[test, [4]], average="macro")),
             float(f1_score(drevo_predict, table.iloc[test, [4]], average="macro"))]
        if sum(ks) > max and sum(ks) != 1:
            kfs = ks
            max = sum(ks)
            test_dats, best_pred_id = table.iloc[test, :], drevo_predict
    [print(f'{k}: {v}') for k, v in dict(zip(['accuracy_score', 'recall_score', 'precision_score', 'f1_score'], map(float, kfs))).items()]
    print(f'average: {sum(ks)/len(ks)}')
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
        ax.scatter(test_dats[x], test_dats[y], test_dats[z], c=best_pred_id, label='Хар-ки цвектов')
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_zlabel(z)
        ax.legend()
        plt.title('Работа классификатора')
        plt.show()

    graph('sepal length (cm)', 'petal length (cm)', 'petal width (cm)')


def lesok(table):
    kfold = KFold(n_splits=3, shuffle=True, random_state=110)
    max = 0
    for train, test in kfold.split(table):
        les_clf = RandomForestClassifier(n_estimators=20, random_state=110)
        les_clf.fit(table.iloc[train, [0, 1, 2, 3]], table.iloc[train, [4]])
        les_predict = les_clf.predict(table.iloc[test, [0, 1, 2, 3]])
        ks = [float(accuracy_score(les_predict, table.iloc[test, [4]])),
              float(recall_score(les_predict, table.iloc[test, [4]], average="macro")),
              float(precision_score(les_predict, table.iloc[test, [4]], average="macro")),
              float(f1_score(les_predict, table.iloc[test, [4]], average="macro"))]
        if sum(ks) > max and sum(ks) != 1:
            kfs = ks
            max = sum(ks)
            test_dats, best_pred_id = table.iloc[test, :], les_predict
    [print(f'{k}: {v}') for k, v in dict(zip(['accuracy_score', 'recall_score', 'precision_score', 'f1_score'], map(float, kfs))).items()]
    print(f'average: {sum(ks) / len(ks)}')

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
        ax.scatter(test_dats[x], test_dats[y], test_dats[z], c=best_pred_id, label='Хар-ки цвектов')
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_zlabel(z)
        ax.legend()
        plt.title('Работа классификатора')
        plt.show()

    graph('sepal length (cm)', 'petal length (cm)', 'petal width (cm)')


def best_param_les(table):
    X_train = table.iloc[:, :-1]
    y_train = table['target']
    clf = RandomForestClassifier()
    parameters = {'n_estimators': range(10, 51, 10),  # число деревьев в лесу. Оно будет изменяться от 10 до 50 с шагом 10
                  'max_depth': range(1, 13, 2),  # глубина дерева. Она будет изменяться от 1 до 12 с шагом в 2
                  'min_samples_leaf': range(1, 8),  # минимальное число образцов в листах. Оно будет изменяться от 1 до 7
                  'min_samples_split': range(2, 10, 2)}  # минимальное число образцов для сплита. Оно будет изменяться от 2 до 9.
    grid = GridSearchCV(clf, parameters, cv=5)
    grid.fit(X_train, y_train)
    print(grid.best_params_)


def best_param_drevo(table):
    X_train = table.iloc[:, :-1]
    y_train = table['target']
    clf = LogisticRegression()
    parameters = {'criterion': ["gini", "entropy"],  # The function to measure the quality of a split
                  'splitter': ["best", "random"],  # The strategy used to choose the split at each node
                  'max_depth': range(1, 13, 2),  # The maximum depth of the tree
                  'min_samples_split': range(2, 10, 2),  # минимальное число образцов для сплита
                  'min_samples_leaf': range(1, 8)}  # минимальное число образцов в листах.
    grid = GridSearchCV(clf, parameters, cv=5)
    grid.fit(X_train, y_train)
    print(grid.best_params_)


def best_param_reg(table):
    X_train = table.iloc[:, :-1]
    y_train = table['target']
    clf = LogisticRegression()
    parameters = {'penalty': ['l1', 'l2', 'elasticnet', 'none'],  #  нормa штрафа
                  'C': np.logspace(-3, 3, 7),  # обратная сила регуляризации; меньшие значения указывают на более сильную регуляризацию.
                  'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],  # алгоритм для использования в задаче оптимизации.
                  'max_iter': range(100, 1001, 100)}  # максимальное количество итераций, необходимое для сходимости решателей.
    grid = GridSearchCV(clf, parameters, cv=5)
    grid.fit(X_train, y_train)
    print(grid.best_params_)


def best_sosedi(table):
    X_train = table.iloc[:, :-1]
    y_train = table['target']
    clf = KNeighborsClassifier()
    parameters = {'n_neighbors': range(1, 10),  # Number of neighbors to use
                  'weights': ['uniform', 'distance'],  # Weight function used in prediction
                  'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],  # Algorithm used to compute the nearest neighbors
                  'metric': ['minkowski', 'euclidean', 'cityblock', 'chebyshev']}  # The distance metric to use for the tree
    grid = GridSearchCV(clf, parameters, cv=5)
    grid.fit(X_train, y_train)
    print(grid.best_params_)

