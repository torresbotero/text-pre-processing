import numpy as np
import itertools

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import GridSearchCV

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import SnowballStemmer
from nltk.corpus import wordnet as wn

import re

import matplotlib.pyplot as plt

from time import time

import xlsxwriter

# Metodos para correr los respectivos experimentos

## Metodo usado para probar y evaluar un clasificador
# El siguiente metodo sirve para evaluar un clasificador enviado como parametro, junto con los datos con que debe ser probado. Las metricas usadas para medir la efectividad del clasificador son la exactitud (accuracy), la precision, la exhaustividad (recall) y el valor-f1 (f1-score). El metodo retorna la matriz de confunsion de los resultados obtenidos.

def test_predictions_and_return_cm(clf, test_data, true_data):
    predicted = clf.predict(test_data)
    
    accuracy = metrics.accuracy_score(true_data, predicted)
    precision, recall, f1_score, support = metrics.precision_recall_fscore_support(true_data, predicted, 
                                                                                   average="macro")
    print("Exactitud:", accuracy)
    print("F1_score:", f1_score)
    print("Precision:", precision)
    print("Recall:", recall)
    print()
    print(metrics.classification_report(true_data, predicted, digits=4))
    cm = metrics.confusion_matrix(true_data, predicted)
    print(cm)
    print()
    return cm

## Metodo para imprimir la matriz de confusion
# El siguiente metodo imprime la matriz de confusion y fue tomado de la documentacion scikit learn. En esta URL se puede encontrar la explicacion y codigo fuente
# http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
# El siguiente metodo "test_nl_classifier" corre el metodo que evalúa el clasificador e imprime la matriz de confusion, es decir, corre los metodos usados para evaluar el clasificador y visualizar los resultados.

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Matriz de confusión',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Clase verdadera')
    plt.xlabel('Clase predicha')
    
def test_nl_classifier(clf, X_t, y_t):
    cm = test_predictions_and_return_cm(clf, X_t, y_t)
    plot_confusion_matrix(cm, classes=['sátiras', 'reales'])

## Metodo para entrenar un clasificador con GridSearchCV
# Este metodo sirve para entrenar un clasificador buscando la mejor combinacion de parametros según el tipo de clasificador. Estos parametros se envían como un objeto diccionario en los argumentos del metodo y se nombran según sea el tipo de clasificador que también se envía en los argumentos del metodo. El metodo siguiente "print_clf_best_params" se usa para mostrar los mejores valores de los parametros obtenidos.

def train_nl_classifier(grid_parameters, clf_type, X, y):
    clf = GridSearchCV(clf_type, grid_parameters, n_jobs=-1, cv=10)
    t0 = time()
    clf.fit(X, y)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)
    return clf

def print_clf_best_params(grid_parameters, clf):
    print("Mejor resultado y parametro encontrados:")
    print("Mejor resultado (accuracy cv): ", clf.best_score_)
    for param_name in sorted(grid_parameters.keys()):
        print("%s: %r" % (param_name, clf.best_params_[param_name]))
    print()
    
## Metodos para vectorizar los textos de entranamiento y prueba
# El metodo fit_train_test_vectors vectoriza los textos de entrenamiento y pruebas y los retorna como vectores de acuerdo al parametro vectorizer. Este metodo es muy importante porque según el parametro vectorizer se representan los textos con algun metodo de ponderacion (tf-idf, tf y binario) y se realizan los procesos de limpiezas indicados al crear el objeto vectorizador. El siguiente metodo es opcional llamarlo y sirve para imprimir las longitudes y formas de los conjuntos resultantes.

def fit_train_test_vectors(vectorizer, X_train, X_test):
    X_train_vector = vectorizer.fit_transform(X_train)
    X_test_vector = vectorizer.transform(X_test)
    return X_train_vector, X_test_vector

def print_train_test_shapes(vectorizer, X_train_vector, X_test_vector):
    print(X_train_vector.shape)
    print()
    
    print(X_test_vector.shape)
    print(vectorizer.get_feature_names()[-150:-100])
    
## Metodos usados para correr las pruebas de linea base y de cada etapa
# Estos son los metodos mas importantes ya que usan todos los metodos explicados anteriormente y sirven para correr los experimentos de forma más rapida y replicable. El primer metodo se debe usar para crear la linea base, es decir, para encontrar los parametros ideales de los clasificadores usados con la linea base (sin ningun proceso de limpieza). El clasificador con mejor resultados puede ser recuperado de la lista de resultados que genera. Al tener el clasificador optimo para la linea base, se procedera a usar los parametros encontrados para hacer los experimentos con cada proceso de limpieza y evaluar que tanto afecta cada proceso por separado. Recibe los parametros vectorizer, grid_parameters y clf_type para ser usados en los metodos "train_nl_classifier" y "fit_train_test_vectors" explicados anteriormente (ver explicacion para entender que son los parametros).


def run_tests_splits(total_data_content, total_data_target, vectorizer, grid_parameters, clf_type):
    results = []
    random_states = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    for state in random_states:
        print("random state: ", state)
        X_train, X_test, y_train, y_test = train_test_split(total_data_content, 
                                                            total_data_target, 
                                                            test_size=0.3, 
                                                            random_state=state)
        X_train_vector, X_test_vector = fit_train_test_vectors(vectorizer, X_train, X_test)
        clf = train_nl_classifier(grid_parameters, clf_type, X_train_vector, y_train)
        y_pred = clf.predict(X_test_vector)
        accuracy = metrics.accuracy_score(y_test, y_pred)
        f1_score = metrics.f1_score(y_test, y_pred, average="macro")
        results.append((clf.best_score_, accuracy, f1_score, X_test_vector, y_test, clf, state, X_train_vector))
    return results

### Metodo para correr un experimento
# De acuerdo a los resultados obtenidos por el metodo anterior, con este metodo se deben correr los experimentos respectivos a cada proceso de limpieza y con los parametros optimos encontrados en la linea base. Esto permite que los resultado obtenidos den una idea de que tanto afecto el proceso de limpieza sin que los parametros del clasificador sean una razon de sesgo o que le quiten validez a los resultados. Recibe los parametros vectorizer, grid_parameters y clf_type para ser usados en los metodos "train_nl_classifier" y "fit_train_test_vectors" explicados anteriormente (ver explicacion para entender que son los parametros).

def run_one_experiment(total_data_content, total_data_target, vectorizer, grid_parameters, clf_type, state):
    X_train, X_test, y_train, y_test = train_test_split(total_data_content, 
                                                        total_data_target, 
                                                        test_size=0.3, 
                                                        random_state=state)
    X_train_vector, X_test_vector = fit_train_test_vectors(vectorizer, X_train, X_test)
    print("Numero de caracteristicas: ", X_train_vector.shape)
    print()
    clf = train_nl_classifier(grid_parameters, clf_type, X_train_vector, y_train)
    y_pred = clf.predict(X_test_vector)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    precision, recall, f1_score, support = metrics.precision_recall_fscore_support(y_test, y_pred, 
                                                                                   average="macro")
    cm = metrics.confusion_matrix(y_test, y_pred)
    true_negatives = cm[0][0]
    true_positives = cm[1][1]
    false_positives = cm[0][1]
    false_negatives = cm[1][0]
    
    features_number = (X_train_vector.shape)[1]
    
    print_clf_best_params(grid_parameters, clf)
    test_nl_classifier(clf, X_test_vector, y_test)
    return (features_number, accuracy,f1_score,precision,recall,true_positives,true_negatives,false_positives,false_negatives)
    
def print_baseline_results(baseline_results, grid_parameters):
    selected_clf_info = max(baseline_results)
    print("Numero de caracteristicas: ", selected_clf_info[7].shape)
    print()
    print("Best score selected: ", selected_clf_info[0])
    print("Best Accuracy: ", selected_clf_info[1])
    print("Best Best F1-score: ", selected_clf_info[2])
    print("Best random state: ", selected_clf_info[6])
    selected_clf = selected_clf_info[5]
    print_clf_best_params(grid_parameters, selected_clf)
    print()
    X_t = selected_clf_info[3]
    y_t = selected_clf_info[4]
    test_nl_classifier(selected_clf, X_t, y_t)
    # for result in tf_baseline_results:
    #     print("Random State: ", result[6])
    #     print("Accuracy: ", result[0])
    #     print("F1-score: ", result[1])
    #     print("Best score: ", result[2])
    #     result_clf = result[5]
    #     print_clf_best_params(parameters_count, result_clf)
    #     print()
    
# Tokenizers(Usados para los diferentes procesos de limpieza)

# Tokenizer por defecto, sin ningun proceso de limpieza
    
def default_tokenize(text):
    regexp_tokenizer = RegexpTokenizer(u'(?u)\\b\\w\\w+\\b')
    tokens =  regexp_tokenizer.tokenize(text)
    return tokens

# Stemming tokenizer

stemmer = SnowballStemmer('spanish')

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize_stems(text):
    regexp_tokenizer = RegexpTokenizer(u'(?u)\\b\\w\\w+\\b')
    tokens =  regexp_tokenizer.tokenize(text)
    token_stems = stem_tokens(tokens, stemmer)
    return token_stems

# Lemmatize tokenizer

# Metodo para hacer el proceso de lematizacion de una palabra
lemmaDict = {}
with open('lemma_data/lemmatization-es.txt', 'rb') as f:
   data = f.read().decode('utf8').replace(u'\r', u'').split(u'\n')
   data = [a.split(u'\t') for a in data]
   
for a in data:
   if len(a) >1:
      lemmaDict[a[1]] = a[0]
   
def lemmatize(word):
   return lemmaDict.get(word, word)
   
def test():
   for a in [ u'salió', u'usuarios', u'abofeteéis', u'diferenciando', u'diferenciándola' ]:
      print(lemmatize(a))
        
def lemmatizer(tokens):
    lemmatized = []
    for item in tokens:
        lemma = lemmatize(item)
        lemmatized.append(lemma)
    return lemmatized

def tokenize_lemmas(text):
    regexp_tokenizer = RegexpTokenizer(u'(?u)\\b\\w\\w+\\b')
    tokens =  regexp_tokenizer.tokenize(text)
    token_lemmas = lemmatizer(tokens)
    return token_lemmas

# Tokenizer para la remocion de URLs

def tokenize_no_urls(text):
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    text_no_urls = re.sub(url_regex, '', text, flags=re.MULTILINE)
    regexp_tokenizer = RegexpTokenizer(u'(?u)\\b\\w\\w+\\b')
    tokens =  regexp_tokenizer.tokenize(text_no_urls)
    return tokens

## Clase para reemplazar las letras repetidas
# El siguiente metodo es tomado y adaptado del libro de Jacob Perkins "Python 3 Text-Processing with NLTK 3 Cookbook". El codigo se puede encontrar en su repositorio de github. https://github.com/japerk/nltk3-cookbook/blob/master/replacers.py. Se adapto para que funcionara con el lenguaje español.

class RepeatReplacer(object):
    """ Removes repeating characters until a valid word is found.
    >>> replacer = RepeatReplacer()
    >>> replacer.replace('looooove')
    'love'
    >>> replacer.replace('oooooh')
    'ooh'
    >>> replacer.replace('goose')
    'goose'
    """
    def __init__(self):
        self.repeat_regexp = re.compile(r'(\w*)(\w)\2(\w*)')
        self.repl = r'\1\2\3'

    def replace(self, word):
        
        # El metodo se modifica en este fragmento para que funcione con el idioma español
        lemma = lemmatize(word)
        
        if wn.synsets(lemma, lang='spa'):
            return word

        repl_word = self.repeat_regexp.sub(self.repl, word)

        if repl_word != word:
            return self.replace(repl_word)
        else:
            return repl_word
        
# Tokenizer para transformar las palabras con letras repetidas

replacer = RepeatReplacer()
def norm_repeated_letters(tokens, replacer):  
    norm_tokens = []
    for item in tokens:
        norm_tokens.append(replacer.replace(item))
    return norm_tokens

def tokenize_norm_letters(text):
    regexp_tokenizer = RegexpTokenizer(u'(?u)\\b\\w\\w+\\b')
    tokens =  regexp_tokenizer.tokenize(text)
    norm_letters = norm_repeated_letters(tokens, replacer)
    return norm_letters


def save_excel_results(file_name, all_experiments_results):
    # Create a workbook and add a worksheet.
    workbook = xlsxwriter.Workbook(file_name)
    worksheet = workbook.add_worksheet()
    row = 0
    col = 0
    for exp in all_experiments_results:
        worksheet.write(row, col, exp[0])
        row += 1
        worksheet.write(row, col, exp[1])
        row += 1
        worksheet.write(row, col, exp[2])
        row += 1
        worksheet.write(row, col, exp[3])
        row += 1
        worksheet.write(row, col, exp[4])
        row += 1
        worksheet.write(row, col, exp[5])
        row += 1
        worksheet.write(row, col, exp[6])
        row += 1
        worksheet.write(row, col, exp[7])
        row += 1
        worksheet.write(row, col, exp[8])
        row += 1
        row += 1
        if row == 30:
            row = 0
            col += 1
    workbook.close()

# Create a workbook and add a worksheet.
def save_excel_comb_results(worksheet, row, exp, optimal_parameters, random_state, comb_name):
    col = 0
    
    worksheet.write(row, col, comb_name)
    row += 1
    worksheet.write(row, col, '# características')
    worksheet.write(row, col + 1, exp[1])
    row += 1
    worksheet.write(row, col, 'Exactitud')
    worksheet.write(row, col + 1, exp[0])
    row += 1
    worksheet.write(row, col, 'Valor-F1')
    worksheet.write(row, col + 1, exp[2])
    row += 1
    worksheet.write(row, col, 'Precisión')
    worksheet.write(row, col + 1, exp[3])
    row += 1
    worksheet.write(row, col, 'Exhaustividad')
    worksheet.write(row, col + 1, exp[4])
    row += 1
    worksheet.write(row, col, 'Verdaderos positivos')
    worksheet.write(row, col + 1, exp[5])
    row += 1
    worksheet.write(row, col, 'Verdaderos negativos')
    worksheet.write(row, col + 1, exp[6])
    row += 1
    worksheet.write(row, col, 'Falsos positivos')
    worksheet.write(row, col + 1, exp[7])
    row += 1
    worksheet.write(row, col, 'Falsos negativos')
    worksheet.write(row, col + 1, exp[8])
    row += 1
    worksheet.write(row, col, 'Parametros')
    worksheet.write(row, col + 1, optimal_parameters)
    row += 1
    worksheet.write(row, col, 'Random state')
    worksheet.write(row, col + 1, random_state)
    row += 1
    row += 1
        
    return row

def run_one_comb_experiment(total_data_content, total_data_target, vectorizer, grid_parameters, clf_type, state):
    X_train, X_test, y_train, y_test = train_test_split(total_data_content, 
                                                        total_data_target, 
                                                        test_size=0.3, 
                                                        random_state=state)
    X_train_vector, X_test_vector = fit_train_test_vectors(vectorizer, X_train, X_test)
    clf = train_nl_classifier(grid_parameters, clf_type, X_train_vector, y_train)
    y_pred = clf.predict(X_test_vector)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    precision, recall, f1_score, support = metrics.precision_recall_fscore_support(y_test, y_pred, 
                                                                                   average="macro")
    cm = metrics.confusion_matrix(y_test, y_pred)
    true_negatives = cm[0][0]
    true_positives = cm[1][1]
    false_positives = cm[0][1]
    false_negatives = cm[1][0]
    
    features_number = (X_train_vector.shape)[1]
    return (accuracy, features_number,f1_score,precision,recall,true_positives,true_negatives,false_positives,false_negatives)
