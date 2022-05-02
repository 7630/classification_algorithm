import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split  # cross_validation
from sklearn.metrics import accuracy_score
from  xgboost import plot_importance
from  matplotlib import pyplot as plt

params = {
    'booster':'gbtree',
    'objective':'multi:softmax',
    'num_class':21,
    'gamma':0.1,
    'max_depth':2,
    'lambda':2,
    'subsample':0.7,
    'colsample_bytree':0.7,
    'min_child_weight':3,
    'eta':0.001,
    'seed':1000,
    'nthread':4,
}

if __name__ == "__main__":
    #以提取电能数据的分类情况为例
    data = pd.read_csv("./e.csv")
    elt = data[['Voltage', 'Current', 'Power', 'Energy', 'Pf']]
    elt_label = data['category']
    elt_label = list(map(int, elt_label))
    train_data, test_data = train_test_split(elt, random_state=1, train_size=0.7, test_size=0.3)
    train_label, test_label = train_test_split(elt_label, random_state=1, train_size=0.7, test_size=0.3)

    data_train = xgb.DMatrix(train_data, label=train_label)
    data_test = xgb.DMatrix(test_data, label=test_label)
    watch_list = [(data_test, 'eval'), (data_train, 'train')]
    param = {'max_depth': 2, 'eta': 0.3, 'silent': 1, 'objective': 'multi:softmax', 'num_class': 21}

    bst = xgb.train(param, data_train, num_boost_round=10, evals=watch_list) #完整参数查看以上params
    y_hat = bst.predict(data_test)

    print('正确率:\t', accuracy_score(test_label,y_hat))
    plot_importance(bst)
    plt.show()
