import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from torch.utils.data import Dataset


def get_data(which=1, normtype=1):
    if which == 1:
        train_data = pd.read_csv('/home/jon/kaggle/tabular-playground-series-feb-2021/train.csv')
        test_data = pd.read_csv('/home/jon/kaggle/tabular-playground-series-feb-2021/test.csv')
        num_names = [x for x in train_data.columns if 'cont' in x]
        cat_names = [x for x in train_data.columns if 'cat' in x]
        ord_names = []
        target = 'target'
        num_classes = 1
    else:
        train_data = pd.read_csv('/home/jon/h2oai-benchmarks/Data/BNPParibas/train.csv')
        test_data = pd.read_csv('/home/jon/h2oai-benchmarks/Data/BNPParibas/test.csv')

        num_names = ['v1', 'v10', 'v100', 'v101', 'v102', 'v103', 'v104', 'v105', 'v106', 'v108', 'v109', 'v11', 'v111', 'v114', 'v115', 'v116', 'v117', 'v118', 'v119', 'v12', 'v120', 'v121', 'v122', 'v123', 'v124', 'v126', 'v127', 'v128', 'v129', 'v13', 'v130', 'v131', 'v14', 'v15', 'v16', 'v17', 'v18', 'v19', 'v2', 'v20', 'v21', 'v23', 'v25', 'v26', 'v27', 'v28', 'v29', 'v32', 'v33', 'v34', 'v35', 'v36', 'v37', 'v38', 'v39', 'v4', 'v40', 'v41', 'v42', 'v43', 'v44', 'v45', 'v46', 'v48', 'v49', 'v5', 'v50', 'v51', 'v53', 'v54', 'v55', 'v57', 'v58', 'v59', 'v6', 'v60', 'v61', 'v62', 'v63', 'v64', 'v65', 'v67', 'v68', 'v69', 'v7', 'v70', 'v72', 'v73', 'v76', 'v77', 'v78', 'v8', 'v80', 'v81', 'v82', 'v83', 'v84', 'v85', 'v86', 'v87', 'v88', 'v89', 'v9', 'v90', 'v92', 'v93', 'v94', 'v95', 'v96', 'v97', 'v98', 'v99']
        num_names = sorted(set(num_names))

        catgen_names = ['v107', 'v110', 'v112', 'v113', 'v125', 'v129', 'v22', 'v24', 'v3', 'v30', 'v31', 'v38', 'v47', 'v52', 'v56', 'v62', 'v66', 'v71', 'v72', 'v74', 'v75', 'v79', 'v91']
        ohe_names = ['v107', 'v110', 'v112', 'v113', 'v129', 'v24', 'v3', 'v30', 'v31', 'v38', 'v47', 'v52', 'v62', 'v66', 'v71', 'v72', 'v74', 'v75', 'v79', 'v91']
        ohe_names += ['v56', 'v125']  # 122 and 90 levels ok
        cat_names = sorted(set(ohe_names))
        cat_names = [x for x in cat_names if x not in num_names]

        ord_names = sorted(set(catgen_names + ohe_names + num_names).difference(num_names + cat_names))

        #num_names = ['v50']
        ord_names = []
        #cat_names = []
        target = 'target'
        idcol = 'ID'
        num_classes = 2

    print("cat_names: %s" % cat_names)
    print("num_names: %s" % num_names)
    print("ord_names: %s" % ord_names)

    # TODO: if ohe, then keep ohe, not numeric
    # TODO: QuantileTransform ohe if numeric before ohe'ing
    # TODO: MICE imputation:
    # https://scikit-learn.org/stable/modules/impute.html#multivariate-feature-imputation
    # https://towardsdatascience.com/whats-the-best-way-to-handle-nan-values-62d50f738fc
    nan = -10

    # NUMERIC
    X_nums = np.vstack([
        train_data[num_names].to_numpy(),
        test_data[num_names].to_numpy()
    ])
    sc = None
    if normtype == 0:
        # no normalization
        pass
    elif normtype == 1:
        X_nums = (X_nums - X_nums.mean(0)) / X_nums.std(0)
    elif normtype == 2:
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
    elif normtype == 3:
        from sklearn.preprocessing import QuantileTransformer
        sc = QuantileTransformer(n_quantiles=2000,
                                 output_distribution='normal',
                                 random_state=42)
    else:
        raise RuntimeError("No such normtype: %s" % normtype)
    if sc is not None:
        X_nums = sc.fit_transform(X_nums)

    # impute out of bounds for neural network, after quantile transformer
    X_nums = np.nan_to_num(X_nums, nan=nan)
    # TODO
    # from sklearn.experimental import enable_iterative_imputer
    # from sklearn.impute import IterativeImputer
    # imp = IterativeImputer(max_iter=10, random_state=0)

    # OHE
    X_cat = np.vstack([
        train_data[cat_names].to_numpy(),
        test_data[cat_names].to_numpy()
    ])
    if X_cat.shape[1] > 0:
        encoder = OneHotEncoder(sparse=False)
        X_cat = encoder.fit_transform(X_cat)

    # High-dimensional CATS
    X_ord = np.vstack([
        train_data[ord_names].to_numpy(),
        test_data[ord_names].to_numpy()
    ])
    if X_ord.shape[1] > 0:
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=nan)
        X_ord = encoder.fit_transform(X_ord)
        X_ord = np.nan_to_num(X_ord, nan=nan).astype(int)

    # STACK
    X = np.hstack([X_cat, X_nums, X_ord])
    y = train_data[target].to_numpy().reshape(-1, 1)
    print("COLS: cats: %s nums: %s ords: %s" % (X_cat.shape[1], X_nums.shape[1], X_ord.shape[1]))
    print("ROWS: cats: %s nums: %s ords: %s y: %s" % (X_cat.shape[0], X_nums.shape[0], X_ord.shape[0], y.shape[0]))


    # Something suboptimal from dae branch
    # epoch  2000 - loss 0.559786 - 20.000000
    # 0.84188

    # normal with simple swap probs, does even better for RMSE:
    # epoch  2000 - loss 0.624798 - 9.000000 sec per epoch0.8412105585745847

    # original before even changed how swap probs set:
    # epoch  1869 - loss 0.606126 - 21.000000 sec per epoch
    # epoch  2000 - loss 0.605735 - 23.000000 sec per epoch0.8412340991691792

    # new 0cbfe03946d2168f2ec76a257df0bb3d3cc1be73
    # epoch  2000 - loss 0.624745 - 20.000000 sec per epoch0.8411460636869601

    if which == 1:
        orig = False
        if orig:
            repeats = [2,  2,  2,  4,  4,  4,  8,  8,  7, 15,  14]
            probas = [.95, .4, .7, .9, .9, .9, .9, .9, .9, .9, .25]
            swap_probas = sum([[p] * r for p, r in zip(probas, repeats)], [])
        else:
            # generic specification also does just fine:
            # /home/jon/Denoise-Transformer-AutoEncoder
            # f3f48918bd93921b8b6777a4fc3fb87d29e6b22b
            # epoch  2000 - loss 0.624798 - 9.000000 sec per epoch0.8412105585745847
            swap_probas = [0.9 if i < X_cat.shape[1] else 0.25 for i in range(X.shape[1])]
    else:
        swap_probs_cat = [0.25] * X_cat.shape[1]
        swap_probs_nums = [0.9] * X_nums.shape[1]
        swap_probs_ord = [0.25] * X_ord.shape[1]
        swap_probas = swap_probs_cat + swap_probs_nums + swap_probs_ord

    assert len(swap_probas) == X.shape[1]

    return X, y, train_data.shape, test_data.shape, \
           len(cat_names), len(num_names), len(ord_names), \
           X_cat.shape[1], X_nums.shape[1], X_ord.shape[1], \
           swap_probas, num_classes


class SingleDataset(Dataset):
    def __init__(self, x, is_sparse=False):
        self.x = x.astype('float32')
        self.is_sparse = is_sparse

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        x = self.x[index]
        if self.is_sparse: x = x.toarray().squeeze()
        return x    
