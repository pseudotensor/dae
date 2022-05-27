import os
import torch
import numpy as np
from datetime import datetime
from util import AverageMeter
from model import SwapNoiseMasker, TransformerAutoEncoder
from data import get_data, SingleDataset
from torch.utils.data import DataLoader
from sklearn.linear_model import Ridge, RidgeClassifier, SGDClassifier
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import mean_squared_error, log_loss

from sklearn.utils.extmath import softmax
#class RidgeClassifierwithProba(RidgeClassifier):
#    def predict_proba(self, X):
#        d = self.decision_function(X)
#        d_2d = np.c_[-d, d]
#        return softmax(d_2d)


def go(which=1, normtype=1, reuse=False, max_epochs=2001, use_xgb=True, xgb_use_gpu=False):
    #  get data
    X, Y, train_shape, test_shape, n_cats_orig, n_nums_orig, n_ords_orig, n_cats, n_nums, n_ords, swap_probas, num_classes = get_data(which=which, normtype=normtype)

    features_file = 'dae_features_which%s_max_epochs%s_n_orig_%s_%s_%s___n_%s_%s_%s__%s_%s.npy' % (which, max_epochs, n_cats_orig, n_nums_orig, n_ords_orig, n_cats, n_nums, n_ords, num_classes, normtype)

    if reuse:
        assert os.path.isfile(features_file), "No such file %s" % features_file
        features = np.load(features_file)
    else:
        features = get_features(X, Y, train_shape, test_shape, n_cats_orig, n_nums_orig, n_ords_orig, n_cats, n_nums, n_ords, swap_probas, num_classes, features_file, max_epochs=max_epochs)

    make_prediction_model(X, Y, train_shape, test_shape, n_cats, n_nums, n_ords, swap_probas, num_classes, features, use_xgb=use_xgb, xgb_use_gpu=xgb_use_gpu)


def make_prediction_model(X, Y, train_shape, test_shape, n_cats, n_nums, n_ords, swap_probas, num_classes, features, use_xgb=True, xgb_use_gpu=False):

    sc = None
    if False and num_classes == 1:
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        Y = sc.fit_transform(Y)

    # downstream supervised regressor
    alpha = 1250  # 1000
    # NOTE: Adds original numeric features
    features = np.concatenate((features, X[:, n_cats:n_cats+n_nums]), axis=1)
    X_train = features[:train_shape[0], :]
    X_test = features[train_shape[0]:, :]

    scores = []
    models = []

    if num_classes == 1:
        train_preds = np.zeros((Y.shape[0], 1))
    else:
        train_preds = np.zeros((Y.shape[0], num_classes))
    test_preds = []

    if num_classes == 1:
        skf = KFold(n_splits=5)#, random_state=42)
        split_gen = skf.split(X_train)
    else:
        skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
        split_gen = skf.split(X_train, Y)

    for train_idx, valid_idx in split_gen:
        train_X = X_train[train_idx]
        train_y = Y[train_idx]
        valid_X = X_train[valid_idx]
        actuals = valid_y = Y[valid_idx]

        if use_xgb:
            model, es = get_xgb_model(xgb_use_gpu=xgb_use_gpu, num_classes=num_classes)
            eval_set = [(valid_X, valid_y)]
            model.fit(train_X, train_y, eval_set=eval_set, eval_metric=model.get_params()['eval_metric'], callbacks=[es])
        else:
            if num_classes == 1:
                model = Ridge(alpha=alpha)
            else:
                model = SGDClassifier(loss="log", n_jobs=20, fit_intercept=True, max_iter=1000, validation_fraction=0.25, early_stopping=True, n_iter_no_change=100)
            model.fit(train_X, train_y)

        models.append(model)
        if num_classes == 2 and not use_xgb:
            print("n_iter_: %s intercept_: %s" % (model.n_iter_, model.intercept_))

        if num_classes == 1:
            preds = model.predict(valid_X)
            test_preds.append(model.predict(X_test))
        else:
            preds = model.predict_proba(valid_X)

            test_preds1 = model.predict_proba(X_test)
            test_preds.append(test_preds1)

        if num_classes == 1:
            score = mean_squared_error(actuals, preds, squared=False)
        else:
            print("actuals/preds size: %s %s" % (list(actuals.shape), list(preds.shape)))
            score = log_loss(actuals.ravel(), preds)
        print("score %s" % score)

        # OOF train preds
        train_preds[valid_idx] = preds

        scores.append(score)

    print(scores)
    print(np.mean(scores))

    # average test preds
    test_preds = np.mean(test_preds)

    np.save('train_preds.npy', train_preds)
    np.save('test_preds.npy', test_preds)


def get_xgb_model(xgb_use_gpu=False, num_classes=1):
    import xgboost as xgb
    num_class = 1 if num_classes <= 2 else num_classes
    if num_classes == 2:
        objective = 'binary:logistic'
        eval_metric = 'logloss'
        maximize = False
    elif num_classes == 1:
        objective = 'reg:squarederror'
        eval_metric = 'rmse'
        maximize = False
    else:
        objective = 'multi:softprob'
        eval_metric = 'mlogloss'
        maximize = False
    early_stopping_rounds = 20
    params = dict(num_class=num_class, objective=objective,
                  early_stopping_rounds=early_stopping_rounds,
                  n_jobs=8,
                  single_precision_histogram=True,
                  eval_metric=eval_metric,
                  n_estimators=200,
                  )
    if xgb_use_gpu:
        params.update(dict(tree_method='gpu_hist', predictor='gpu_predictor', gpu_id=0))
    model = xgb.XGBClassifier(**params)

    es = xgb.callback.EarlyStopping(
        rounds=early_stopping_rounds,
        #abs_tol=1e-3,
        min_delta=1e-3,
        save_best=True,
        maximize=maximize,
        data_name="validation_0",
        metric_name=eval_metric,
    )

    return model, es


def get_features(X, Y, train_shape, test_shape, n_cats_orig, n_nums_orig, n_ords_orig, n_cats, n_nums, n_ords, swap_probas, num_classes, features_file, max_epochs=2001):
    # Hyper-params
    model_params = dict(
        hidden_size=1024,
        num_subspaces=8,
        embed_dim=128,
        num_heads=8,
        dropout=0,
        feedforward_dim=512,
        emphasis=.75,
        mask_loss_weight=2
    )
    batch_size = 384
    init_lr = 3e-4
    lr_decay = .998

    train_dl = DataLoader(
        dataset=SingleDataset(X),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True
    )

    # setup model
    model = TransformerAutoEncoder(
        num_inputs=X.shape[1],
        n_cats_orig=n_cats_orig,
        n_nums_orig=n_nums_orig,
        n_ords_orig=n_ords_orig,
        n_cats=n_cats,
        n_nums=n_nums,
        n_ords=n_ords,
        num_classes=num_classes,
        **model_params
    ).cuda()
    model_checkpoint = 'model_checkpoint.pth'

    print(model)

    noise_maker = SwapNoiseMasker(swap_probas)
    optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay)

    # train model
    for epoch in range(max_epochs):
        t0 = datetime.now()
        model.train()
        meter = AverageMeter()
        for i, x in enumerate(train_dl):
            x = x.cuda()
            x_corrputed, mask = noise_maker.apply(x)
            optimizer.zero_grad()
            loss = model.loss(x_corrputed, x, mask)
            loss.backward()
            optimizer.step()

            meter.update(loss.detach().cpu().numpy())

        delta = (datetime.now() - t0).seconds
        scheduler.step()
        print('\r epoch {:5d} - loss {:.6f} - {:4.6f} sec per epoch'.format(epoch, meter.avg, delta), end='')

    torch.save({
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "model": model.state_dict()
    }, model_checkpoint
    )
    model_state = torch.load(model_checkpoint)
    model.load_state_dict(model_state['model'])

    # extract features
    dl = DataLoader(dataset=SingleDataset(X), batch_size=1024, shuffle=False, pin_memory=True, drop_last=False)
    features = []
    model.eval()
    with torch.no_grad():
        for x in dl:
            features.append(model.feature(x.cuda()).detach().cpu().numpy())
    features = np.vstack(features)
    np.save(features_file, features)

    return features


if __name__ == "__main__":
    #go(which=2, normtype=1, reuse=False, max_epochs=2000, use_xgb=True, xgb_use_gpu=False)
    go(which=2, normtype=1, reuse=True, max_epochs=2000, use_xgb=True, xgb_use_gpu=False)
