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
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import seaborn as sb

iris = load_iris()
# 0 - iris_setosa, 1- iris_versicolor, 2-iris_virginica
# 'sepal length (cm)' ,  'petal length (cm)', 'petal width (cm)', 'sepal width (cm)'
iris_pd = pd.DataFrame(data=np.c_[iris['data'], iris['target']], columns=iris['feature_names'] + ['target'])
chemic_bd = pd.read_csv('Chemical_process.csv', delimiter=';', dtype='float64')
baro_bd = pd.read_csv('dataset_barotrop.csv', delimiter=',')
baro_bd = baro_bd.replace({'Class': {'BARO': 0, 'TROP': 1}})


def pca(tabl):
    features = ['YIELD', 'HCL', 'NH3', 'H20', 'CATALYST']
    x = tabl.loc[:, features].values
    x = StandardScaler().fit_transform(x) # pd.DataFrame(data=x, columns=features).head())
    pca = PCA(n_components=2)
    prcp_compon = pca.fit_transform(x)
    prcp_df = pd.DataFrame(data=prcp_compon, columns=['PC1', 'PC2'])
    final_df = pd.concat([prcp_df], axis=1)
    model = KMeans(n_clusters=5, random_state=100, max_iter=10000)
    model.fit(final_df)
    all_predict = model.predict(final_df)
    fig = plt.figure()
    tabl['YIELD'] = all_predict
    '''for i in range(5):
        tab = tabl.loc[table.YIELD == i, :]
        ax.scatter(tab[x], tab[y], tab[z])
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.legend()
        plt.show()
    print(all_predict)'''
    print(tabl.head(10))


def tsne(table):
    features = ['Longitude', 'Latitude']
    X = table.loc[:, features].values
    cvetochki = TSNE(n_components=2, perplexity=3, early_exaggeration=12, n_iter=5000, random_state=100)
    X_cvetochki = pd.DataFrame(cvetochki.fit_transform(X))
    plt.scatter(X_cvetochki[0], X_cvetochki[1], label='Измерения')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.legend()
    plt.title('t-SNE')
    plt.show()


def k_sosedi(table):
    kfold = KFold(n_splits=3, shuffle=True,  random_state=215436)
    max = 0
    for train, test in kfold.split(table):
        knn_mod = KNeighborsClassifier(n_neighbors=6, algorithm='auto', metric='chebyshev', weights='distance')
        knn_mod.fit(table.iloc[train, [0, 1]], table.iloc[train, [2]])
        knn_pred = knn_mod.predict(table.iloc[:, [0, 1]])
        ks = [float(accuracy_score(knn_pred, table.iloc[:, [2]])),
            float(recall_score(knn_pred, table.iloc[:, [2]], average="macro")),
            float(precision_score(knn_pred, table.iloc[:, [2]], average="macro")),
            float(f1_score(knn_pred, table.iloc[:, [2]], average="macro"))]
        if sum(ks) > max and sum(ks) != 1:
            kfs = ks
            max = sum(ks)
            test_dats, best_pred_id = table.iloc[:, :], knn_pred
    [print(f'{k}: {v}') for k, v in dict(zip(['accuracy_score', 'recall_score', 'precision_score', 'f1_score'], map(float, kfs))).items()]
    print(f'average: {sum(ks)/len(ks)}')

    def graph(x, y):
        fig = plt.figure()
        ax = fig.add_subplot(1, 2, 1)
        ax.scatter(test_dats[x], test_dats[y], c=test_dats.Class, label='Хар-ки ')
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.legend()
        plt.title('Реальные классы')
        ax = fig.add_subplot(1, 2, 2)
        ax.scatter(test_dats[x], test_dats[y], c=best_pred_id, label='Хар-ки ')
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.legend()
        plt.title('Работа классификатора')
        plt.show()
    graph('Longitude', 'Latitude')



def lin_reg_class(table):
    kfold = KFold(n_splits=3, shuffle=True, random_state=110)
    max = 0
    for train, test in kfold.split(table):
        lreg_clf = LogisticRegression(max_iter=100, C=0.001, penalty='l1', solver='liblinear')
        lreg_clf.fit(table.iloc[train, [0, 1]], table.iloc[train, [2]])
        lrg_predict = lreg_clf.predict(table.iloc[test, [0, 1]])
        ks = [float(accuracy_score(lrg_predict, table.iloc[test, [2]])),
             float(recall_score(lrg_predict, table.iloc[test, [2]], average="macro")),
             float(precision_score(lrg_predict, table.iloc[test, [2]], average="macro")),
             float(f1_score(lrg_predict, table.iloc[test, [2]], average="macro"))]
        if sum(ks) > max and sum(ks) != 1:
            kfs = ks
            max = sum(ks)
            test_dats, best_pred_id = table.iloc[test, :], lrg_predict
    [print(f'{k}: {v}') for k, v in dict(zip(['accuracy_score', 'recall_score', 'precision_score', 'f1_score'], map(float, kfs))).items()]
    print(f'average: {sum(ks)/len(ks)}')

    def graph(x, y, z):
        fig = plt.figure()
        ax = fig.add_subplot(1, 2, 1)
        ax.scatter(test_dats[x], test_dats[y], c=test_dats.target, label='Хар-ки ')
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.legend()
        plt.title('Реальные классы')
        ax = fig.add_subplot(1, 2, 2)
        ax.scatter(test_dats[x], test_dats[y], c=best_pred_id, label='Хар-ки цвектов')
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.legend()
        plt.title('Работа классификатора')
        plt.show()



def drevo(table):
    kfold = KFold(n_splits=3, shuffle=True, random_state=110)
    max = 0
    for train, test in kfold.split(table):
        drevo_clf = DecisionTreeClassifier(criterion='gini', max_depth=7, min_samples_leaf=4, min_samples_split=9,
                                           splitter='random', random_state=110)
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
    kfold = KFold(n_splits=3, shuffle=True, random_state=215436)
    max = 0
    for train, test in kfold.split(table):
        les_clf = RandomForestClassifier(n_estimators=30, random_state=110, max_depth=5, min_samples_leaf=2,
                                         min_samples_split=2)
        les_clf.fit(table.iloc[train, [0, 1]], table.iloc[train, [2]])
        les_predict = les_clf.predict(table.iloc[test, [0, 1]])
        ks = [float(accuracy_score(les_predict, table.iloc[test, [2]])),
              float(recall_score(les_predict, table.iloc[test, [2]], average="macro")),
              float(precision_score(les_predict, table.iloc[test, [2]], average="macro")),
              float(f1_score(les_predict, table.iloc[test, [2]], average="macro"))]
        if sum(ks) > max and sum(ks) != 1:
            kfs = ks
            max = sum(ks)
            test_dats, best_pred_id = table.iloc[test, :], les_predict
    [print(f'{k}: {v}') for k, v in dict(zip(['accuracy_score', 'recall_score', 'precision_score', 'f1_score'], map(float, kfs))).items()]
    print(f'average: {sum(ks) / len(ks)}')

    def graph(x, y):
        fig = plt.figure()
        ax = fig.add_subplot(1, 2, 1)
        ax.scatter(test_dats[x], test_dats[y], c=test_dats.Class, label='Хар-ки ')
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.legend()
        plt.title('Реальные классы')
        ax = fig.add_subplot(1, 2, 2)
        ax.scatter(test_dats[x], test_dats[y], c=best_pred_id, label='Хар-ки ')
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.legend()
        plt.title('Работа классификатора')
        plt.show()
    graph('Longitude', 'Latitude')


def best_param_les(table):
    X_train = table.iloc[:, :-1]
    y_train = table['Class']
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
    clf = DecisionTreeClassifier()
    parameters = {'criterion': ["gini", "entropy"],  # The function to measure the quality of a split
                  'splitter': ["best", "random"],  # The strategy used to choose the split at each node
                  'max_depth': range(1, 13),  # The maximum depth of the tree
                  'min_samples_split': range(2, 12),  # минимальное число образцов для сплита
                  'min_samples_leaf': range(1, 8)}  # минимальное число образцов в листах.
    grid = GridSearchCV(clf, parameters, cv=5)
    grid.fit(X_train, y_train)
    print(grid.best_params_)


def best_param_reg(table):
    X_train = table.iloc[:, :-1]
    y_train = table['Class']
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
    y_train = table['Class']
    clf = KNeighborsClassifier()
    parameters = {'n_neighbors': range(1, 10),  # Number of neighbors to use
                  'weights': ['uniform', 'distance'],  # Weight function used in prediction
                  'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],  # Algorithm used to compute the nearest neighbors
                  'metric': ['minkowski', 'euclidean', 'cityblock', 'chebyshev']}  # The distance metric to use for the tree
    grid = GridSearchCV(clf, parameters, cv=5)
    grid.fit(X_train, y_train)
    print(grid.best_params_)


def himikk():
    X = chemic_bd.drop('YIELD', axis=1)
    y = chemic_bd['YIELD']
    bp = RandomForestRegressor()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=110)
    model = RandomForestRegressor(n_estimators=10, oob_score=True, random_state=110)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(f'MSE: {mean_squared_error(y_test, pred)}')
    print(f'MAE: {mean_absolute_error(y_test, pred)}')
    sb.boxplot(data=chemic_bd)
    plt.show()
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.scatter(X_test['HCL'], X_test['NH3'], y_test, label='Измерения')
    ax.set_xlabel('HCL')
    ax.set_ylabel('NH3')
    ax.set_zlabel('YIELD')
    ax.set_title('Измерения реальные')
    ax.legend()
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.scatter(X_test['HCL'], X_test['NH3'], pred, label='Измерения')
    ax.set_xlabel('HCL')
    ax.set_ylabel('NH3')
    ax.set_zlabel('YIELD')
    ax.set_title('Измерения предсказанные')
    ax.legend()
    plt.show()


def razvedka():
    plt.subplot(1, 2, 1)
    plt.scatter(baro_bd['Longitude'], baro_bd['Latitude'], c=baro_bd.Class)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Исходное распределение')
    plt.subplot(1, 2, 2)
    sb.boxplot(baro_bd.drop('Class', axis=1))
    plt.title('Box-plot')
    plt.show()

