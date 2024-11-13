class SimpleModel:
    def __init__(self):
        self.mode = {}
    
    def fit(self, X, y):
        #create a dictionary to store the count of the y_vals for each X
        count_dict = {}
        for x_val, y_val in zip(X, y):
            if x_val not in count_dict:
                count_dict[x_val] = {}
            if y_val not in count_dict[x_val]:
                count_dict[x_val][y_val] = 0
            count_dict[x_val][y_val] += 1

        
        # Determine the most frequent y value for each X
        self.mode = {}
        for x_val, y_counts in count_dict.items():
            most_frequent_y = None
            max_count = -1
            #iterate over the x_val[y_val] counts to find the most frequent y
            # Iterate over counts to find the most frequent y
            for y_val, count in y_counts.items():
                 # Choose the most frequent y or the last one in case of a tie
                if count > max_count or (count == max_count and y_val > most_frequent_y):
                    max_count = count
                    most_frequent_y = y_val
                self.mode[x_val] = most_frequent_y
    def predict(self, X):
        #use the self.mode dictionary to predict the y_val for each x in x test
        predictions = [self.mode.get(x_val, None) for x_val in X]
        return predictions

def train_and_predict(X_train, y_train, X_test):
    model = SimpleModel()
    #fit the model with x_train and y_train
    model.fit(X_train, y_train)
    #predict the labels of x_test
    predictions = model.predict(X_test)
    return predictions
def distances_option_calculator(X, y, distance_type):
    if distance_type == "euclidian":
        return (sum([(X[i] - y[i]) ** 2 for i in range(len(X))])) **0.5
    elif distance_type == "manhattan":
        return sum([abs(X[i] - y[i]) for i in range(len(X))])
    elif distance_type == "hamming":
        return [(X[i] != y[i] for i in range(len(X)))]
import numpy as np
import numpy as np
class KNN:
    def __init__(self, k):
        self.points = None
        self.labels = None
        self.k = k

    def distance(self, point_a, point_b):
        #calculate the euclidean distance between two points
        return np.sqrt(np.sum((np.array(point_a) - np(array(point_b)) ** 2)))
    
    def fit(self, X_train, y_train):
        self.points = X_train
        self.labels = y_train



    def predict(self, X_test):
        predictions = []
        for test_
        predictions = []
        for test_point in X_test:
            distances = []
            for i, train_point in enumerate(self.points):
                dist = self.distance(test_point, train_point)
                #store the distance and the corresponding label
                distances.append(dist, self.labels[i])



            # Sort distances and get the k nearest labels
            distances.sort(key=lambda x: x[0])
            k_nearest_labels = [label for _, label in distances[:self.k]]
            # Predict the most common label among the k nearest neighbors
            predicted_label = max(set(k_nearest_labels), key=k_nearest_labels.count)
            predictions.append(predicted_label)
        return predictions

def euclidian_distance(point_a, point_b):
    # Calculate the Euclidean distance between two points
    return (sum([(point_a[i] - point_b[i])**2 for i in range(len(point_a))]))**0.5

def wcss(points):
    num_points = len(points)
    centroid = [sum(point[i] for point in points) / num_points for i in range(len(points[0]))]

   sum_distances = sum(euaclidean_distance(point, centroid) for point in points)
   #Step 2: Calculate the sum of Euclidean distances to the centroid
   return sum_distances / num_points

import numpy as np
def euclidian_distance(point_a, point_b):
    return (sum([(point_a[i] - point_b[i])**2 for i in range(len(point_a))]))**0.5

class KMeans:
    def __init__(self, k):
        self.k = k
        self.centroids = None

    def find_cluster_centroid(self, cluster_points):
        num_points = len(cluster_points)
        if num_points == 0:
            return np.zeros(len(cluster_points[0]))
        centroid = [sum(point[i] for point in cluster_points) / num_points for i in range(len(cluster_points[0]))]
        return centroid

    def fit(self, X_train):
        self.centroids = [X_train[i] for i in range(2, len(X_train), 2)[:self.k]]
        
        prev_centroids = None
        while True:
            clusters = [[] for _ in range(self.k)]
            for point in X_train:
                distances = [euclidian_distance(point, centroid) for centroid in self.centroids]
                closest_centroid_idx = np.argmin(distances)
                clusters[closest_centroid_idx].append(point)
            
            new_centroids = [self.find_cluster_centroid(cluster) for cluster in clusters]
            
            if prev_centroids is not None:
                total_movement = sum(euclidian_distance(new_centroids[i], prev_centroids[i]) for i in range(self.k))
                if total_movement < 0.01:
                    break
            
            prev_centroids = self.centroids
            self.centroids = new_centroids
    
    def predict(self, X_test):
        predictions = []
        for point in X_test:
            distances = [euclidian_distance(point, centroid) for centroid in self.centroids]
            closest_centroid_idx = np.argmin(distances)
            predictions.append(closest_centroid_idx)
        return predictions

import pandas as pd
from collections import defaultdict

class NaiveBayes:
    def __init__(self):
        self.probability_features = {}
        self.probabilty_target = {}
        self.classes = []
    
    def fit(self, X_train, y_train):
        self.probability_features = {}
        self.probability_target = {}
        self.classes = y_train.unique()
        target_variable = "target"
        df = X_train.copy()
        df[target_variable] = y_train

        distinct_target_values = df[target_variable].unique()
        for column in df.columns:
            if column == target_variable:
                continue
            self.probability_features[column] = {}
            distinct_values = df[column].unique()
            for value in distinct_values:
                self.probability_features[column][value] = {}
                for target_value in distinct_target_values: 
                    total = len(df[(df[column] == value) & (df[target_variable] == target_value)][target_variable])
                    count = len(df[(df[column] == value)])
                    self.probability_features[column][value][target_value] = total / count #will hold 1 / 2

        for target_value in distinct_target_values: 
            total = len(df[(df[target_variable] == target_value)][target_variable])
            #both yes and no are target values and play is the target variable
            self.probability_target[target_value] = total / len(df[target_variable])
    def get_classes_probability(self, X):
        #X holds a dataframe with a single row
        classes_probabilities = {}
        for cls in self.classes:
            classes_probabilities[cls] = 1
            for column in X.columns:
                for distinct_value in X[column].unique():
                    #this multiplies the value of that feauture e.g low stored in the self.probability_features with 1 eg 1/2 * 1(low = play = yes = 1/2)
                    classes_probabilities[cls] *= self.probability_feautures[column][distinct_value][cls]
            #that value is now multiplied by the targets value probabilty with respect to the cls e.g[yes(2/3)]
            classes_probabilities[cls] *= self.probability_target[cls]
        return classes_probabilities
    def predict(self, X_test):
        predictions = []
        #iterate through the rows
        for _, row in X_test.itterrows():
            row_df = pd.DataFrame([row])
            #calculate the value for each row in X_test
            classes_probabilities = self.get_classes_probability(row_df)
            #hold the maximum between no and yes
            predicted_class = max(class_probabilities, key=class_probabilities.get)
            predictions.append(predicted_class)
        return predictions

import numpy as np
def sigmoid(point):
    arr = np.array(point)
    # Write code here
    sigmoid_values = 1 / (1 + np.exp(-arr))
    return sigmoid_values
def linear_equation(coef, x):
    res = coef[0]
    for i in range(len(x)):
        res += coef[i+1]*x[i]
    return res

def decide_outcomes(rules, outcomes, data_point):
    current_rule = "rule1"
    output = None
    while current_rule in outcomes:
        data_value = data_point[current_rule]
        rule_type = rules[current_rule]["type"]
        rule_value = rules[current_rule]["value"]
        if rule_type == "bigger";
            res = data_value > rule_value
        else:
            res = data_value < rule_value
    current_rule = outcomes[current_rule][res]
    if current_rule is None:
        return output
    else:
        output = current_rule
    return current_rule/outout
        


