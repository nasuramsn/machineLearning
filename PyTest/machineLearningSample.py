'''
Created on 2017/07/26

@author: Akimitsu Hirose
'''

import numpy as np
import timeit
import scipy as sp
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from scipy.optimize import fsolve

def error(f, x, y):
    return sp.sum((f(x)-y) ** 2)


#IrisがSetosaかそれ以外か判定する
def apply_model(example):
    if example[2] < 2: print("Iris Setosa")
    else: print("Iris Virginica or Iris Versicolor")


def chapter1():
    a = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    # 配列を出力
    #print(a)
    # 次元数を出力
    #print(a.ndim)
    # 要素数を出力
    #print(a.shape)

    #aを再配列
    b = a.reshape(4,2)
    b[0][1] = 77
    # 配列を出力
    #print(b)
    # 次元数を出力
    #print(b.ndim)
    # 要素数を出力
    #print(b.shape)

    c = a.reshape((4,2)).copy()
    c[0][0] = 99
    #print(c)
    #print(a)

    #print(a[np.array([2,4,6])])
    #print(a > 5)
    #print(a.clip(0, 4))

    #処理速度比較
    #normal_py_sec = timeit.timeit('sum(x*x for x in xrange(1000))', number = 10000)
    #good_num_sec = timeit.timeit('na.dot(na)', setup="import numpy as np; na=np.arange(1000)", number=10000)
    #print("normal_py f% sec:"%normal_py_sec)
    #print("good_num f% sec:"%good_num_sec)

    #データファイル読込み
    data = sp.genfromtxt("web_traffic.tsv", delimiter="\t")
    #print(data[:10])
    #print(data.shape)

    x = data[:,0]
    y = data[:,1]

    x=x[~sp.isnan(y)]
    y=y[~sp.isnan(y)]

    #グラフ描画
    plt.scatter(x, y)
    plt.title("web traffic research")
    plt.xlabel("time")
    plt.ylabel("hits/hour")
    plt.xticks([w*7*24 for w in range(10)], ['week %i' %w for w in range(10)])
    plt.autoscale(tight = True)
    plt.grid()
    #plt.show()

    #誤差情報
    fp1, residuals, rank, sv, rcond =sp.polyfit(x, y, 1, full=True)
    #print("Model Parameters %s:" % fp1)
    #print(residuals)

    #近似線描画
    #f1 = sp.poly1d(fp1)
    #fx = sp.linspace(0, x[-1], 1000)
    #plt.plot(fx, f1(fx), linewidth=4)
    #plt.legend(["d=%i" % f1.order], loc="upper left")
    #plt.show()

    #変化点を3.5週目に置く
    inflection = 3 * 7 * 24
    xbf = x[:inflection]
    ybf = y[:inflection]
    xaf = x[inflection:]
    yaf = y[inflection:]

    fbf = sp.poly1d(sp.polyfit(xbf, ybf, 1))
    faf = sp.poly1d(sp.polyfit(xaf, yaf, 1))

    fbf_error = error(fbf, xbf, ybf)
    faf_error = error(faf, xaf, yaf)

    #print("Error Infrection %f" % (fbf_error + faf_error))

    #適切な近似式を求める
    frac = 0.3  #テストに用いるデータの割合
    split_idx = int(frac*len(xaf))

    shuffled = sp.random.permutation(list(range(len(xaf))))    #xafの30%のデータを取得する
    test = sorted(shuffled[:split_idx])                                         #テスト用データ
    train = sorted(shuffled[split_idx:])                                        #訓練用データ

    #各々訓練用データで訓練を行う
    #fbt1 = sp.poly1d(sp.polyfit(xaf[train], yaf[train], 1))
    fbt2 = sp.poly1d(sp.polyfit(xaf[train], yaf[train], 2))
    #fbt3 = sp.poly1d(sp.polyfit(xaf[train], yaf[train], 3))
    #fbt10 = sp.poly1d(sp.polyfit(xaf[train], yaf[train], 10))
    #fbt100 = sp.poly1d(sp.polyfit(xaf[train], yaf[train], 100))

    #それぞれのテスト結果の評価を行う
    #for f in [fbt1, fbt2, fbt3, fbt10, fbt100]:
    #    print("Error  d=%i: %f" % (f.order, error(f, xaf[test], yaf[test])))

    #fbt2を使用して、100,000リクエスト/hを超える日を算出
    print(fbt2)

    reached_max = fsolve(fbt2-100000, 800)/(7*24)
    print("result %f" % reached_max[0])


def chapter2():
    data = load_iris()

    features = data['data']
    feature_name = data['feature_names']
    target = data['target']
    target_names = data['target_names']
    labels = target_names[target]

    #for t, marker, c in zip(range(3), ">ox", "rgb"):
        #plt.scatter(features[target == t,0], features[target == t,1], marker=marker, c=c)
    #plt.show()

    #花弁の長さによる判定
    plength = features[:, 2]
    is_setosa = (labels == 'setosa')    #setosaかどうかの判定配列

    max_setosa = plength[is_setosa].max()
    min_non_setosa = plength[~is_setosa].min()
    features_no_setosa = features[~is_setosa]
    labels_no_setosa = labels[~is_setosa]
    labels_virginica = (labels_no_setosa == 'virginica')
    #print('Max of setosa: {0}.' .format(max_setosa))
    #print('Min of others: {0}.' .format(min_non_setosa))

    #判定処理でSetosaかどうか確認する
    for t in features:
        apply_model(t)

if __name__ == '__main__':
    chapter2()