import numpy as np
from bisect import bisect_left
from typing import List, Optional

import matplotlib.pyplot as plt
from scipy.stats import beta

from grazie.common.main.ranking.ranker import RankResult


def binom_interval(success, total, err=0.05):
    quantile = err / 2.
    lower = beta.ppf(quantile, success, total - success + 1)
    upper = beta.ppf(1 - quantile, success + 1, total - success)
    return round(lower, 4), round(upper, 4)


def precision(qs: List[RankResult], k: Optional[int] = 1):
    cnt, ok_cnt = 0, 0
    for rank_res in qs:
        target = [sugg.target for sugg in rank_res.variants]
        rand = np.random.random(len(target))
        pts = list(zip(rank_res.scores, rand, target))
        res = sorted(pts, reverse=True)
        if k is not None:
            res = res[:k]
        else:
            res = sorted(pts, key=lambda x: (x[0], x[2]), reverse=True)
        cnt += 1
        ok_cnt += int(1 in set(x[2] for x in res))
    ci = binom_interval(ok_cnt, cnt)
    return round(ok_cnt / cnt, 4), ci


def precision_recall_balance(qs: List[RankResult], quantiles: List[float] = None):
    quantiles = quantiles or []
    y_true, y_pred = [], []
    for rank_res in qs:
        targets = [sugg.target for sugg in rank_res.variants]
        score, target = sorted(zip(rank_res.scores, targets), reverse=True)[0]
        y_true.append(target)
        y_pred.append(score)

    thrs = sorted(list(set(y_pred)), reverse=True)

    pairs = sorted(zip(y_pred, y_true), reverse=True)
    idx, true_cnt = 0, 0.0
    true_cnts, shows = [], []
    for i, th in enumerate(thrs):
        while idx < len(pairs) and pairs[idx][0] >= th:
            true_cnt += pairs[idx][1]
            idx += 1
        true_cnts.append(true_cnt)
        shows.append(idx)

    accs = [1.0] + [true_cnt / idx for true_cnt, idx in zip(true_cnts, shows)]
    shows_ratio = [0.0] + [idx / len(pairs) for idx in shows]
    sorted_accs = sorted(accs)
    if quantiles:
        print("Accuracy, Shows, Threshold")
        for quantile in quantiles:
            ind = len(accs) - bisect_left(sorted_accs, quantile) - 1
            q_accs = round(accs[min(ind, len(accs) - 1)], 3)
            q_shows = round(shows_ratio[min(ind, len(shows_ratio) - 1)], 3)
            th = round(thrs[min(ind, len(thrs) - 1)], 3)
            print(q_accs, q_shows, th)

    plt.plot(shows_ratio, accs)
    plt.xlabel("Shown ration")
    plt.ylabel("Accuracy in shown")
    plt.show()

    return accs[-1]
