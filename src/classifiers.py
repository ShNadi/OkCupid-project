from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from imblearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
plt.rc("font", size=10)
import itertools
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import FeatureUnion
from sklearn import preprocessing
import matplotlib
import pandas as pd


# plot confusion matrix
def plot_confusion_matrix(cm, folder, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0])
        , range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig("../data/output/"+ folder + "/ConfusionMatrix.png")
    plt.close()


# Show most informative words in logistic regression classifier
def show_most_informative_features(vectorizer, clf, folder, n=50):
    feature_names = vectorizer.get_feature_names()
    coefs_with_fns = sorted(zip(clf.coef_[0], feature_names))
    top = zip(coefs_with_fns[:n], coefs_with_fns[:-(n + 1):-1])
    for (coef_1, fn_1), (coef_2, fn_2) in top:
        with open(r"../data/output/"+ folder+"/FeatureImportance.txt", 'a+') as output:
            output.write("\t%.4f\t%-15s\t\t%.4f\t%-15s\n" % (coef_1, fn_1, coef_2, fn_2))


# Text classification by Logistic regression(Bag of words)
def logistic_text(clean_text, target):

    X_train, X_val, y_train, y_val = train_test_split(clean_text['clean_text'], target['isced'], stratify=target, test_size = 0.25, random_state=0)

    clf_text = Pipeline([('vec', CountVectorizer(max_df=0.60, max_features=200000, stop_words='english', binary=True, lowercase=True, ngram_range=(1, 2))),
                    ('clf', LogisticRegression(random_state=0, max_iter=10000, solver='lbfgs', penalty='l2', class_weight='balanced'))])
    clf_text.fit(X_train, y_train)

    predictions_t = clf_text.predict(X_val)

    print("Final Accuracy for Logistic: %s"% accuracy_score(y_val, predictions_t))
    cm = confusion_matrix(y_val, predictions_t)
    print(classification_report(y_val, predictions_t))
    with open (r"../data/output/BagOfWords/classificationReport.txt", 'a+') as output:
        output.write(classification_report(y_val, predictions_t))

    plot_confusion_matrix(cm, folder="BagOfWords", classes=[0, 1], normalize=False,
                          title='Confusion Matrix')

    # Show most informative words
    show_most_informative_features(clf_text.get_params()['vec'], clf_text.get_params()['clf'], n=50,
                                   folder="BagOfWords")
    print("Outputs are written in 'data/output/BagOfWords' folder.\n")



# The Logistic regression(Text+ Language features)
def logistic_text_lingustic(text_meta, target):

    X_train, X_val, y_train, y_val = train_test_split(text_meta, target['isced'], stratify=target, test_size = 0.25,
                                                      random_state=0)

    cols = text_meta.loc[:, text_meta.columns != 'clean_text'].columns

    get_text_data = FunctionTransformer(lambda x: x['clean_text'], validate=False)
    get_numeric_data = FunctionTransformer(lambda x: x[cols], validate=False)

    process_and_join_features = Pipeline([
        ('features', FeatureUnion([
            ('numeric_features', Pipeline([
                ('selector', get_numeric_data),
                ('scaler', preprocessing.StandardScaler())

            ])),
            ('text_features', Pipeline([
                ('selector', get_text_data),
                ('vec', CountVectorizer(binary=False, ngram_range=(1, 2), lowercase=True))
            ]))
        ])),
        ('clf',
         LogisticRegression(random_state=0, max_iter=10000, solver='lbfgs', penalty='l2', class_weight='balanced'))
    ])

    # merge vectorized text data and scaled numeric data
    process_and_join_features.fit(X_train, y_train)
    predictions_tm = process_and_join_features.predict(X_val)

    print("Final Accuracy for Logistic: %s" % accuracy_score(y_val, predictions_tm))
    cm = confusion_matrix(y_val, predictions_tm)
    print(classification_report(y_val, predictions_tm))

    with open(r"../data/output/TextLinguistic/classificationReport.txt", 'a+') as output:
        output.write(classification_report(y_val, predictions_tm))

    plot_confusion_matrix(cm, folder="TextLinguistic", classes=[0, 1], normalize=False,
                          title='Confusion Matrix')

    show_most_informative_features(
        process_and_join_features.get_params()['features'].get_params()['text_features'].get_params()['vec'],
        process_and_join_features.get_params()['clf'], n=50, folder='TextLinguistic')

    print("Outputs are written in 'data/output/TextLinguistic' folder.\n")


# The Logistic regression (LIWC only)
def logistic_liwc(liwc, target):
    X_train, X_val, y_train, y_val = train_test_split(liwc, target['isced'], stratify=target, test_size=0.25,
                                                      random_state=0)


    scaler = preprocessing.StandardScaler()
    Xl_train_scaled = scaler.fit_transform(X_train)
    Xl_val_scaled = scaler.transform(X_val)

    LogisticRegr = LogisticRegression(random_state=0, max_iter=10000, solver='lbfgs', penalty='l2',
                                      class_weight='balanced')
    LogisticRegr.fit(Xl_train_scaled, y_train)
    predictions = LogisticRegr.predict(Xl_val_scaled)

    print("Final Accuracy for Logistic: %s" % accuracy_score(y_val, predictions))

    cm = confusion_matrix(y_val, predictions)
    plot_confusion_matrix(cm, folder="Liwc", classes=[0, 1], normalize=False,
                          title='Confusion Matrix')

    print(classification_report(y_val, predictions))
    with open(r"../data/output/Liwc/classificationReport.txt", 'a+') as out:
        out.write(classification_report(y_val, predictions))


    # Plot feature importance for Liwc
    coef_liwc = pd.Series(LogisticRegr.coef_[0], index=liwc.columns)

    imp_coef_liwc = coef_liwc.sort_values()
    matplotlib.rcParams['figure.figsize'] = (8.0, 15.0)
    imp_coef_liwc.plot(kind="barh")
    plt.title("Feature importance ")
    plt.savefig("../data/output/liwc/FeatureImportance.png")
    plt.show()


# The Logistic regression (Bag of words + LIWC)
def logistic_text_liwc(liwc_text, target):
    X_train, X_val, y_train, y_val = train_test_split(liwc_text, target['isced'], stratify=target, test_size=0.25,
                                                      random_state=0)

    cols = liwc_text.loc[:, liwc_text.columns != 'clean_text'].columns

    get_text_data = FunctionTransformer(lambda x: x['clean_text'], validate=False)
    get_numeric_data = FunctionTransformer(lambda x: x[cols], validate=False)

    process_and_join_features = Pipeline([
        ('features', FeatureUnion([
            ('numeric_features', Pipeline([
                ('selector', get_numeric_data),
                ('scaler', preprocessing.StandardScaler())

            ])),
            ('text_features', Pipeline([
                ('selector', get_text_data),
                ('vec', CountVectorizer(binary=False, ngram_range=(1, 2), lowercase=True))

            ]))
        ])),
        (
        'clf', LogisticRegression(random_state=0, max_iter=5000, solver='lbfgs', penalty='l2', class_weight='balanced'))
    ])


    # merge vectorized text data and scaled numeric data
    process_and_join_features.fit(X_train, y_train)
    predictions_lt = process_and_join_features.predict(X_val)

    print("Final Accuracy for Logistic: %s" % accuracy_score(y_val, predictions_lt))
    cm = confusion_matrix(y_val, predictions_lt)
    plt.figure()
    plot_confusion_matrix(cm, folder="TextLiwc", classes=[0, 1], normalize=False,
                          title='Confusion Matrix')
    print(classification_report(y_val, predictions_lt))
    with open(r"../data/output/TextLiwc/classificationReport.txt", 'a+') as out:
        out.write(classification_report(y_val, predictions_lt))


    # Write_most_informative_features
    n = 1000
    fnames = dict(process_and_join_features.named_steps['features'].transformer_list).get('text_features').named_steps[
        'vec'].get_feature_names()
    k = liwc_text.columns.tolist()
    k.remove(k[93])
    names = k + fnames
    dd = pd.DataFrame(columns=names)
    feature_names = dd.columns
    clff = process_and_join_features['clf']
    coefs_with_fns = sorted(zip(clff.coef_[0], feature_names))
    top = zip(coefs_with_fns[:n], coefs_with_fns[:-(n + 1):-1])
    for (coef_1, fn_1), (coef_2, fn_2) in top:
        with open(r"../data/output/TextLiwc/FeatureImportance.txt", "a") as output:
            output.write("\t%.4f\t%-15s\t\t%.4f\t%-15s\n" % (coef_1, fn_1, coef_2, fn_2))

