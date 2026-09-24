# Laboratory practice 2.2: KNN classification
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme()
import numpy as np  
import seaborn as sns


def minkowski_distance(a, b, p=2):
    """
    Compute the Minkowski distance between two arrays.

    Args:
        a (np.ndarray): First array.
        b (np.ndarray): Second array.
        p (int, optional): The degree of the Minkowski distance. Defaults to 2 (Euclidean distance).

    Returns:
        float: Minkowski distance between arrays a and b.
    """
    
    """
    Versión primera:
    c = a-b #resta los dos arrays
    c = np.abs(c) #hace valor absoluto de cada elemento del array numpy
    c = np.power(c,p) #c**p eleva cada valor del array c al valor p
    c = np.sum(c) #suma todos los valores del array c
    c = np.power(c,1/p) #eleva el valor c a la potencia 1/p
    """
    return (np.sum(np.abs(a-b)**p))**(1/p) #optimización 

# k-Nearest Neighbors Model

# - [K-Nearest Neighbours](https://scikit-learn.org/stable/modules/neighbors.html#classification)
# - [KNeighborsClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)


class knn:
    def __init__(self):
        self.k = None
        self.p = None
        self.x_train = None
        self.y_train = None
        

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, k: int = 5, p: int = 2):
        """
        Fit the model using X as training data and y as target values.

        You should check that all the arguments shall have valid values:
            X and y have the same number of rows.
            k is a positive integer.
            p is a positive integer.

        Args:
            X_train (np.ndarray): Training data.
            y_train (np.ndarray): Target values.
            k (int, optional): Number of neighbors to use. Defaults to 5.
            p (int, optional): The degree of the Minkowski distance. Defaults to 2.
        """
        if len(X_train) != len(y_train):
            raise ValueError("Length of X_train and y_train must be equal.")
        elif k <= 0 or p <= 0:
            raise ValueError("k and p must be positive integers.")
        else:  
            self.x_train = X_train
            self.y_train = y_train
            self.k = k
            self.p = p
        
          

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the class labels for the provided data.

        Args:
            X (np.ndarray): data samples to predict their labels.

        Returns:
            np.ndarray: Predicted class labels.
        """
        """
        Explicación:
        En predict_proba devolvemos las probabilides de cada output, de forma que es un soft clasifier
        ya que podemos usar umbrales para tener diferentes fronteras de decisión.
        
        Aquí, usamos un hard classifier, devolvemos el output mayoritario directamente, sin probabilidad
        asociada.
        
        La idea es la misma que la de la función siguiente (explicada más en detalle) solo que aquí
        usamos la función auxiliar most_common_label para obtener la clase del vecino mayoritario y 
        eso es lo que devolvemos, sin probabilidad asociada.
        """
        
        
        predicted_class = []
        for dato in X:
            distances = self.compute_distances(dato)
            vecinos_index = self.get_k_nearest_neighbors(distances)
            vecinos_clases = self.y_train[vecinos_index]
            predicted_class.append(self.most_common_label(vecinos_clases))
        
        return np.array(predicted_class)

    def predict_proba(self, X):
        """
        Predict the class probabilities for the provided data.

        Each class probability is the amount of each label from the k nearest neighbors
        divided by k.

        Args:
            X (np.ndarray): data samples to predict their labels.

        Returns:
            np.ndarray: Predicted class probabilities.
        """
        """
        Explicación:
        
        Nos entran tres puntso que queremos predecir su Y:
        X = [[x1,y1],[x2,y2],[x3,y3]]
        
        --> Primero calculamos, para cada punto que queremos obtener su predicción, su distancia respecto todos los puntos
        distance_matrix = [distancia_primerpunto_por_predecir_con_todos_los_datos, distancia_segundopunto_por_predecir_con_todos_los_datos...]
        distancia_matrix = [[5,4,3],[5,4,6],[5,4,8]]
        
        -->Luego obtenemos, por cada nuevo dato que queremos predecir, los índices de sus K vecinos más cercanos usando la función auxiliar
        (usando k definido en fit). En este ejemplo, con k=1, quedaría:
        get_index_k__neighbours = [[2],[1],[1]]
        
        --> Posteriormente, extraemos la clase de esos vecinos (recordar que distancia es sobre X pero para obtener probabilidad hay que 
        contar los votos en Y entonces self.y_train[index_eighbours]. Quedaría por ejemplo:
        get_k_class = [[0],[1],[0]] 
        Por ejemplo, para el primer dato que queremos  predecir, hemos evaluado su vecino más cercano y tiene clase 0 (NO), por tanto
        la probabilidad p(Y=NO|X)=1 y p(Y=1|X)= 0.
        
        -->Calculamos las probabilidades de cada clase (aquí es fácil porque solo tienen un vecino) y devolvemos una lista de listas
        con [[p(Y=0|X),p(Y=1|X)],...]
        
        Toda esta explicación la he hecho previamente cuando lo hacía de maner poco óptima. Ahora
        la idea final es la misma pero pongo el for fuera y así no trabajo con matrices grandes en 
        cada paso.
        """
        
        get_prob = []
        for dato in X:
            dist = self.compute_distances(dato)
            vecinos_index = self.get_k_nearest_neighbors(dist)
            vecinos_clases = self.y_train[vecinos_index]
            p_y_1 = len(vecinos_clases[vecinos_clases=="YES"]) / len(vecinos_clases)
            p_y_0 = 1 - p_y_1 
            
            get_prob.append([p_y_0,p_y_1])
                        
        return np.array(get_prob) #me daba error porque no era array de numpy y en el test hacen .shape
    
    def compute_distances(self, point: np.ndarray) -> np.ndarray:
        """Compute distance from a point to every point in the training dataset

        Args:
            point (np.ndarray): data sample.

        Returns:
            np.ndarray: distance from point to each point in the training dataset.
        """
        """
        Explicación:
        Vamos recorriendo los datos y calculando la distancia minkowski del nuevo dato
        con cada uno de ellos usando la función creada al principio.
        Devolvemos un array que representa la distancia del nuevo input con cada dato 
        asociado a esa posición en x_train.
        """
        return np.array([minkowski_distance(dato,point,self.p) for dato in self.x_train])

    def get_k_nearest_neighbors(self, distances: np.ndarray) -> np.ndarray:
        """Get the k nearest neighbors indices given the distances matrix from a point.

        Args:
            distances (np.ndarray): distances matrix from a point whose neighbors want to be identified.

        Returns:
            np.ndarray: row indices from the k nearest neighbors.

        Hint:
            You might want to check the np.argsort function.
        """
        """
        Explicación:
        Nos llega una lista con las distancias del nuevo dato a cada punto asociado con esa
        posición del array en x_train. [2 , 1 , 3 , 5]
        np.argsort te ordena la lista en orden ASCENDENTE y te devuelve una lista con los índices
        originales de los elementos ordenados.
        Ejemplo: la lista ordenada sería [1,2,3,5] y argsort devuelve [1,0,2,3]
        
        Por último nos quedamos con los k primeros haciendo un slicing de esa lista de
        índices representando las distancias de menor a mayor.
        Ejemplo: con k = 2:  [1,0]
        """
        return  np.argsort(distances)[:self.k] 

    def most_common_label(self, knn_labels: np.ndarray) -> int:
        """Obtain the most common label from the labels of the k nearest neighbors

        Args:
            knn_labels (np.ndarray): labels from the k nearest neighbors

        Returns:
            int: most common label
        """
        
        """
        Explicación
        c = np.unique_counts(array con 0s y 1s) [0,0,1,1,1]
        c.values = [0,1] #te devuelve array con las clases no repetidas
        c.values = [2,3] #te devuelve array con las repeticiones por clases
        np.argmax(array) #te devuelve el índice del valor máximo del array
        
        Código inicial:
        uniq = np.unique_counts(knn_labels)
        categorias = uniq.values #array con las distintas clases [0,1]
        i_repeticiones = np.argmax(uniq.counts) #array con las repeticiones de esas clases [2,3]
        
        return categorias[i_repeticiones]
        """
        
        #Código optimizado
        uniq = np.unique_counts(knn_labels)
        return uniq.values[np.argmax(uniq.counts)]
    

    def __str__(self):
        """
        String representation of the kNN model.
        """
        return f"kNN model (k={self.k}, p={self.p})"



def plot_2Dmodel_predictions(X, y, model, grid_points_n):
    """
    Plot the classification results and predicted probabilities of a model on a 2D grid.

    This function creates two plots:
    1. A classification results plot showing True Positives, False Positives, False Negatives, and True Negatives.
    2. A predicted probabilities plot showing the probability predictions with level curves for each 0.1 increment.

    Args:
        X (np.ndarray): The input data, a 2D array of shape (n_samples, 2), where each row represents a sample and each column represents a feature.
        y (np.ndarray): The true labels, a 1D array of length n_samples.
        model (classifier): A trained classification model with 'predict' and 'predict_proba' methods. The model should be compatible with the input data 'X'.
        grid_points_n (int): The number of points in the grid along each axis. This determines the resolution of the plots.

    Returns:
        None: This function does not return any value. It displays two plots.

    Note:
        - This function assumes binary classification and that the model's 'predict_proba' method returns probabilities for the positive class in the second column.
    """
    
    # Map string labels to numeric
    unique_labels = np.unique(y)
    num_to_label = {i: label for i, label in enumerate(unique_labels)}

    # Predict on input data
    preds = model.predict(X)
    
    # Determine TP, FP, FN, TN
    tp = (y == unique_labels[1]) & (preds == unique_labels[1])
    fp = (y == unique_labels[0]) & (preds == unique_labels[1])
    fn = (y == unique_labels[1]) & (preds == unique_labels[0])
    tn = (y == unique_labels[0]) & (preds == unique_labels[0])

    # Plotting
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # Classification Results Plot
    ax[0].scatter(X[tp, 0], X[tp, 1], color="green", label=f"True {num_to_label[1]}")
    ax[0].scatter(X[fp, 0], X[fp, 1], color="red", label=f"False {num_to_label[1]}")
    ax[0].scatter(X[fn, 0], X[fn, 1], color="blue", label=f"False {num_to_label[0]}")
    ax[0].scatter(X[tn, 0], X[tn, 1], color="orange", label=f"True {num_to_label[0]}")
    ax[0].set_title("Classification Results")
    ax[0].legend()

    # Create a mesh grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_points_n),
        np.linspace(y_min, y_max, grid_points_n),
    )

    # # Predict on mesh grid
    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = model.predict_proba(grid)[:, 1].reshape(xx.shape)

    # Use Seaborn for the scatter plot
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette="Set1", ax=ax[1])
    ax[1].set_title("Classes and Estimated Probability Contour Lines")

    # Plot contour lines for probabilities
    cnt = ax[1].contour(xx, yy, probs, levels=np.arange(0, 1.1, 0.1), colors="black")
    ax[1].clabel(cnt, inline=True, fontsize=8)

    # Show the plot
    plt.tight_layout()
    plt.show()



def evaluate_classification_metrics(y_true, y_pred, positive_label):
    """
    Calculate various evaluation metrics for a classification model.

    Args:
        y_true (array-like): True labels of the data.
        positive_label: The label considered as the positive class.
        y_pred (array-like): Predicted labels by the model.

    Returns:
        dict: A dictionary containing various evaluation metrics.

    Metrics Calculated:
        - Confusion Matrix: [TN, FP, FN, TP]
        - Accuracy: (TP + TN) / (TP + TN + FP + FN)
        - Precision: TP / (TP + FP)
        - Recall (Sensitivity): TP / (TP + FN)
        - Specificity: TN / (TN + FP)
        - F1 Score: 2 * (Precision * Recall) / (Precision + Recall)
    """
    
    """
    Explicación:
    Para obtener los valores de True Positive y True Negative, tenemos que comparar
    los outputs predichos (y_pred_mapped) y los reales (y_true_mapped)  y quedarnos 
    con los que coinciden ya que corresponden a los aciertos de nuestro modelo.
    
    Para ello, primero nos quedamos con el array de valores predichos pero aplicando
    un filtro que hace que nos quedemos solo con los valores de dicho array que coinciden
    con el mismo valor en y_true_mapped. Con esto tenemos una lista de 1s y 0s que
    corresponden a los valores acertados por nuestro modelo (todos los 0 acertados y
    los 1 acertados).
    [0,0,1,1,1,0]
    
    Luego queremos contar, para cada clase, cuantos valores tenemos para saber cúantos 
    True Positive (1) y True Negative (0) hemos acertado. Para ello, usamos la función
    np.bincount([0,0,1,1,1,0]) que devuelve un array con las ocurrencias de cada clase
    ordenadas por el valor de cada clase (el conteo de la clase 0 irá antes en el array
    que el conteo de la clase 1). Obteniendo con este ejemplo:[3,3]= TN,TP
    De este modo tenemos los TP y TN. Para sacar los casos de FP,FN tenemos que hacer 
    lo mismo pero cambiando el filtro para quedarnos con los no-coincidencias entre 
    el array de valores predichos y array de valores reales.
    
    Con todo esto tenemos todo listo para las operaciones. Es importante añadir en los 
    cálculos una condición para evitar problemas de divisiones por 0:
    if (tp+fp) else 0.0
    
    Con esto tendremos toda esta función terminada
    """
    # Map string labels to 0 or 1
    y_true_mapped = np.array([1 if label == positive_label else 0 for label in y_true])
    y_pred_mapped = np.array([1 if label == positive_label else 0 for label in y_pred])

    # Confusion Matrix
    tn,tp = np.bincount(y_pred_mapped[y_true_mapped==y_pred_mapped],minlength=2) #devuelve un array con conteo de apariciones de cada clase, con minlength devuelve array de dos posiciones aunque no cuente ninguno de alguna clase
    fn,fp = np.bincount(y_pred_mapped[y_true_mapped!=y_pred_mapped],minlength=2)
    
    # Accuracy
    accuracy = (tp+tn)/(tp+tn+fn+fp)

    # Precision
    precision = tp/(tp+fp) if (tp+fp) else 0.0

    # Recall (Sensitivity)
    recall = tp/(tp+fn) if (tp+fn) else 0.0

    # Specificity
    specificity = tn/(tn+fp) if (tn+fp) else 0.0

    # F1 Score
    f1 = 2*(precision*recall)/(precision+recall) if (precision+recall) else 0.0

    return {
        "Confusion Matrix": [tn, fp, fn, tp],
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "Specificity": specificity,
        "F1 Score": f1,
    }


def plot_calibration_curve(y_true, y_probs, positive_label, n_bins=10):
    """
    Construye una calibration curve:
      - bin_centers: centros fijos de cada bin (midpoints)
      - true_proportions: proporción real de positivos dentro de cada bin

    Devuelve un dict con esas dos arrays.
    """
    y_true_mapped = np.array([1 if label == positive_label else 0 for label in y_true])
    
    bordes = np.linspace(0,1,n_bins+1) #creamos n+1 cortes entre 0 y 1 (obtenemos n zonas, siempre hace falta un corte más que zonas)
    
    centro_bines = (bordes[:-1] + bordes[1:])/2
    
    """
    linspace no está dando por ejemplo [0,0.2,0.4,0.6,0.8,1] (6 bordes, 5 zonas)
    bordes[:-1] : nos quedamos con los bordes izquierdos [0,0.2,0.4,0.6,0.8]
    bordes[1:] : nos quedamos con los bordes izquierdos [0.2,0.4,0.6,0.8,1]
    Luego divido por dos y tenemos el valor central de cada bin.
    [0.1,0.3,0.5,0.7,0.9]
    """
    
    indices_bins = np.digitize(y_probs, bordes[1:-1],right=False)
    
    """
    np.digitize(x, bins) te devuelve en qué bin cae cada x.
    
    Pero digitize no espera los bordes completos con 0 y 1.
    Por ello le pasamos [0.2,0.4,0.6,0.8] y digitiza ya 
    genera índices para cada zona. (0 si la prob cae por 
    debajo de 0.2, 1 si la prob cae por debajo de 0.4...)
    
    Luego usamos right=False para que cada bin sea de tipo 
    [a,b).
    
    Obtendremos entonces una lista que nos dirá para cada
    posición en qué grupo cae esa probabilidad.
    
    [0.1,0.5,0.3,0.7] --> [0,2,1,3]
    
    """
    
    proporciones_reales = np.empty(n_bins, dtype=float)
    proporciones_reales[:] = np.nan
    
    """
    Creamos un array de longitud n_bins y lo rellenamos
    entero con Nan. Lo hacemos así por eficiencia ya que 
    es más costoso hacerlo con ceros e ir cambiando
    """
    
    for k in range(n_bins):
        mask = (indices_bins == k) #nos quedamos con matriz booleana con True para las probabilidades pertenecientes a esa clase
        suma = mask.sum()  #obtenemos cúantos elementos hay en ese bin
        if suma > 0:
            proporciones_reales[k] = y_true_mapped[mask].mean() #obtenemos la proporción de positivos reales en ese bin
    
    """
    Vamos entonces recorriendo los bins
    Creamos una máscara booleana que devuelve True para las posiciones de las 
    probabilidades predichas que pertenecen a ese bin
    
    Con el ejemplo de antes: para el bin 0 [0,0.2) :  [True, False, False, False]
    Tiene sentido, las probs son [0.1,0.5,0.3,0.7] --> La única del bin 0 es la primera
    
    Con esa máscara booleana, obtenemos los valores reales predichos para
    las probabilidades de cada bin y vemos la media de valores positivos predichos    
    """
    
    true_proportions = proporciones_reales
    bin_centers = centro_bines
    
    
    calibration_dict = {"bin_centers": bin_centers, "true_proportions": true_proportions}
    x = calibration_dict["bin_centers"]
    y = calibration_dict["true_proportions"]

    plt.figure()
    plt.plot([0, 1], [0, 1], linestyle="--")   # línea ideal y=x
    plt.plot(x, y, marker="o")

    plt.xlabel("Predicted probability (bin center)")
    plt.ylabel("True positive rate")
    plt.title("Calibration Curve")

    plt.xlim(0, 1)
    plt.ylim(0, 1)

    plt.show()
    
    return calibration_dict



def plot_probability_histograms(y_true, y_probs, positive_label, n_bins=10):
    """
    Plot probability histograms for the positive and negative classes separately.

    This function creates two histograms showing the distribution of predicted
    probabilities for each class. This helps in understanding how the model
    differentiates between the classes.

    Args:
        y_true (array-like): True labels of the data. Can be binary or categorical.
        y_probs (array-like): Predicted probabilities for the positive class. 
                            Expected values are in the range [0, 1].
        positive_label (int or str): The label considered as the positive class.
                                    Used to map categorical labels to binary outcomes.
        n_bins (int, optional): Number of bins for the histograms. Defaults to 10. 
                                Bins are equally spaced in the range [0, 1].

    Returns:
        dict: A dictionary with the following keys:
            - "array_passed_to_histogram_of_positive_class": 
                Array of predicted probabilities for the positive class.
            - "array_passed_to_histogram_of_negative_class": 
                Array of predicted probabilities for the negative class.

    """
   
    y_true = np.asarray(y_true)
    y_probs = np.asarray(y_probs, dtype=float)

    y_true_mapped = (y_true == positive_label).astype(int)

    probs_pos = y_probs[y_true_mapped == 1]
    probs_neg = y_probs[y_true_mapped == 0]
    
    # ---- PLOTEO ----
    # bins igualmente espaciados entre 0 y 1
    bins = np.linspace(0, 1, n_bins + 1)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    ax[0].hist(probs_pos, bins=bins, edgecolor="black")
    ax[0].set_title("Histogram (positive class)")
    ax[0].set_xlabel("Predicted probability")
    ax[0].set_ylabel("Count")
    ax[0].set_xlim(0, 1)

    ax[1].hist(probs_neg, bins=bins, edgecolor="black")
    ax[1].set_title("Histogram (negative class)")
    ax[1].set_xlabel("Predicted probability")
    ax[1].set_xlim(0, 1)

    plt.tight_layout()
    plt.show()
    # ----------------

    return {
        "array_passed_to_histogram_of_positive_class": y_probs[y_true_mapped == 1],
        "array_passed_to_histogram_of_negative_class": y_probs[y_true_mapped == 0],
    }



def plot_roc_curve(y_true, y_probs, positive_label):
    """
    Plot the Receiver Operating Characteristic (ROC) curve.

    The ROC curve is a graphical representation of the diagnostic ability of a binary
    classifier system as its discrimination threshold is varied. It plots the True Positive
    Rate (TPR) against the False Positive Rate (FPR) at various threshold settings.

    Args:
        y_true (array-like): True labels of the data. Can be binary or categorical.
        y_probs (array-like): Predicted probabilities for the positive class. 
                            Expected values are in the range [0, 1].
        positive_label (int or str): The label considered as the positive class.
                                    Used to map categorical labels to binary outcomes.

    Returns:
        dict: A dictionary containing the following:
            - "fpr": Array of False Positive Rates for each threshold.
            - "tpr": Array of True Positive Rates for each threshold.

    """
    y_true = np.asarray(y_true)
    y_probs = np.asarray(y_probs, dtype=float)

    y_true_bin = (y_true == positive_label).astype(int)
    P = np.sum(y_true_bin == 1)
    N = np.sum(y_true_bin == 0)

    thresholds = np.linspace(0, 1, 11)

    fpr = np.zeros(len(thresholds), dtype=float)
    tpr = np.zeros(len(thresholds), dtype=float)

    for i, thr in enumerate(thresholds):
        y_pred = (y_probs >= thr).astype(int)

        TP = np.sum((y_pred == 1) & (y_true_bin == 1))
        FP = np.sum((y_pred == 1) & (y_true_bin == 0))

        tpr[i] = TP / P if P > 0 else 0.0
        fpr[i] = FP / N if N > 0 else 0.0

    # plot (da igual para el test, pero lo pide el enunciado)
    plt.figure()
    plt.plot(fpr, tpr, marker="o")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.show()

    return {"fpr": fpr, "tpr": tpr}
