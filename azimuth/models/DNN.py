import numpy as np
import scipy as sp
import sklearn

def DNN_on_fold(feature_sets, train, test, y, y_all, X, dim, dimsum, learn_options):
    import theanets
    from sklearn.metrics import accuracy_score
    y = np.array(y_all[learn_options['DNN target variable']].values, dtype=float)
    y_train, X_train = y[train][:, None], X[train]
    y_test, X_test = y[test][:, None], X[test]

    num_hidden_layers = [1]#, 2, 3]
    num_units = [2]#, 5, 8, 10, 15, 20, 25, 30, 40, 50, 60]
    accuracies = np.zeros((len(num_hidden_layers), len(num_units)))
    best_score = None
    best_model = None

    for i, hl in enumerate(num_hidden_layers):
        for j, nu in enumerate(num_units):
            architecture = np.zeros((2+hl,))
            architecture[0] = X_train.shape[1]
            architecture[-1] = 1#len(np.unique(y_train))
            architecture[1:-1] = [nu for l in range(hl)]

            if learn_options["cv"] == "stratified":
                label_encoder = sklearn.preprocessing.LabelEncoder()
                label_encoder.fit(y_all['Target gene'].values[train])
                gene_classes = label_encoder.transform(y_all['Target gene'].values[train])
                n_folds = len(np.unique(gene_classes))
                cv = sklearn.cross_validation.StratifiedKFold(gene_classes, n_folds=n_folds, shuffle=True)
            elif learn_options["cv"]=="gene":
                gene_list = np.unique(y_all['Target gene'].values[train])
                cv = []
                for gene in gene_list:
                    cv.append(get_train_test(gene, y_all[train]))
                n_folds = len(cv)


            for train_ind, valid_ind in cv:
                # e = theanets.Experiment(
                #     theanets.Classifier,
                #     layers=architecture,
                #     train_batches=32,
                #     # patience=100,
                #     # tied_weights=False,
                #     )

                e = theanets.Experiment(
                    theanets.Regressor,
                    layers=architecture,
                    train_batches=32,
                    # patience=100,
                    # tied_weights=False,
                    )

                e.run((X_train[train_ind], y_train[train_ind]), (X_train[valid_ind], y_train[valid_ind]))
                pred = e.network.predict(X_train[valid_ind])

                accuracies[i, j] += sp.stats.spearmanr(pred.flatten(), y_train[valid_ind].flatten())[0]

            accuracies[i, j] = accuracies[i, j]/float(n_folds)


            if best_score is None or accuracies[i, j] > best_score:
                best_score = accuracies[i, j]
                best_model = copy.deepcopy(e)
                print "DNN with %d hidden layers and %d units, accuracy: %.4f   *" % (hl, nu, accuracies[i,j])
            else:
                print "DNN with %d hidden layers and %d units, accuracy: %.4f" % (hl, nu, accuracies[i,j])

    best_model.run((X_train, y_train), (X_test, y_test))
    y_pred = best_model.network.predict(X[test])

    return y_pred, None