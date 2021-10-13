from typing import Optional, List, Tuple

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from grazie.common.main.ranking.evaluation import precision
from grazie.common.main.ranking.ranker import Ranker, RankVariant, RankQuery, ranked_queries


class NeuralRanker(nn.Module, Ranker):
    def __init__(self, dims: List[int] = None):
        super().__init__()
        self.dims = dims or []
        self.seq = nn.Sequential()
        self.sig = nn.Sigmoid()

    def forward(self, input_1, input_2):
        s1 = self.seq(input_1)
        s2 = self.seq(input_2)
        out = self.sig(s2 - s1)
        return torch.cat((1 - out, out)).view(-1, 2)

    def predict(self, s: List[List[float]]) -> List[float]:
        with torch.no_grad():
            out = self.seq(torch.FloatTensor(s))
            return out.numpy()[:, 0].tolist()

    def importance_info(self, train_data: List[RankQuery]):
        print("Features importance")
        for name, imp in enumerate(list(self.parameters())[0].data.numpy()[0]):
            print(name, imp)
        print()

    def _init_model(self, train_data: List[RankQuery]):
        input_dim = len(train_data[0].variants[0].features)

        layers: List[nn.Module] = []
        pred_dim = input_dim
        for i, dim in enumerate(self.dims):
            layers.append(nn.Linear(pred_dim, dim))
            layers.append(nn.Sigmoid())
            pred_dim = dim
        layers.append(nn.Linear(pred_dim, 1))

        self.seq = nn.Sequential(*layers)

    def fit(self, train_data: List[RankQuery], test_data: List[RankQuery],
            epochs=1, lr=3e-2, l2=0., l1=0., **kwargs) -> 'NeuralRanker':
        self._init_model(train_data)

        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=l2)
        loss_function = nn.CrossEntropyLoss()
        for epoch in range(epochs):
            inds = np.random.permutation(list(range(len(train_data))))
            for i, idx in enumerate(inds):
                optimizer.zero_grad()
                suggs = train_data[idx].variants
                loss = self._get_loss(loss_function, suggs)

                if loss is not None:
                    loss.backward()
                    optimizer.step()

            test_res = self._test_info(test_data, loss_function)
            train_res = self._test_info(train_data, loss_function)
            print(f"Epoch {epoch}\n Train {train_res}\n Test {test_res}")

        return self

    def _get_loss(self, loss_function, suggs: List[RankVariant]) -> Optional[torch.Tensor]:
        bad, good = split_query(suggs)
        if bad:
            target = torch.tensor([1] * len(bad))
            bad_tensor = torch.FloatTensor(bad)
            good_tensor = torch.FloatTensor(good)
            pred = self(bad_tensor, good_tensor)
            loss = loss_function(pred, target)
            return loss
        return None

    def _test_info(self, test_data: List[RankQuery], loss_function):
        with torch.no_grad():
            loss_val = torch.tensor(0.0)
            for query in test_data:
                loss = self._get_loss(loss_function, query.variants)
                if loss is not None:
                    loss_val += loss.data
            ranked_qs = ranked_queries(test_data, self)
            prec_1, ci_1 = precision(ranked_qs, k=1)
            prec_3, ci_3 = precision(ranked_qs, k=3)
            return f"Loss {loss_val / len(test_data)}, Prec@1 {prec_1}: {ci_1}, Prec@3 {prec_3}: {ci_3}"

    def save(self, path: str):
        torch.save(self.seq, path)

    def load(self, path: str) -> 'NeuralRanker':
        self.seq = torch.load(path)
        return self


def split_query(suggs: List[RankVariant]) -> Tuple[List[List[float]], List[List[float]]]:
    max_target = max([s.target for s in suggs])
    bad, good = [], []
    ts = []
    for i, sugg1 in enumerate(suggs):
        for j, sugg2 in enumerate(suggs[i:]):
            if i < j:
                if sugg1.target == max_target and sugg1.target > sugg2.target:
                    bad.append(sugg2.features)
                    good.append(sugg1.features)
                    ts.append((sugg2.target, sugg1.target))
                if sugg2.target == max_target and sugg2.target > sugg1.target:
                    bad.append(sugg1.features)
                    good.append(sugg2.features)
                    ts.append((sugg1.target, sugg2.target))
            # if i < j:
            #     if sugg1.target < sugg2.target:
            #         bad.append(sugg1.features)
            #         good.append(sugg2.features)
            #         ts.append((sugg1.target, sugg2.target))
            #     elif sugg1.target > sugg2.target:
            #         good.append(sugg1.features)
            #         bad.append(sugg2.features)
            #         ts.append((sugg2.target, sugg1.target))

    # fs_ind: Dict[float, List[List[float]]] = {0.0: [], 1.0: []}
    # for sugg in suggs:
    #     fs_ind[sugg.target].append(sugg.features)
    # bad, good = [], []
    # for fs1 in fs_ind[0]:
    #     for fs2 in fs_ind[1]:
    #         bad.append(fs1)
    #         good.append(fs2)
    return bad, good
