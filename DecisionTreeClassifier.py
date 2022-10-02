import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

def train_test_split(X, y, test_size=0.3, random_state=None, shuffle=True):
    ''' 
    Split the datasets into train data and test data.
    '''
    if shuffle:
        X, y = shuffleData(X, y, random_state)
  
    split_idx = int(len(y) - (len(y) // (1/test_size)))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    return X_train, X_test, y_train, y_test
  
def shuffleData(X, y, random_state=None):
    '''
    Random shuffle data from the given samples.
    '''
    if random_state:
        np.random.seed(random_state)
    idxs = np.arange(X.shape[0])
    np.random.shuffle(idxs)

    return X.to_numpy()[idxs], y.to_numpy()[idxs]

def accuracy_score(y_true, y_pred):
    """ 
    Compare y_true to y_pred and return the accuracy 
    """
    accuracy = np.sum(y_true == y_pred, axis=0) / len(y_true)
    return accuracy

def cross_val_score(model, X, y, scoring='accuracy', *, cv):
    '''
    Function to calculate the K-Fold Cross Validation Score.
    '''

    scores = []

    for train_idx, test_idx in cv.split(X):
        X_train_val, X_test_val = X.to_numpy()[train_idx], X.to_numpy()[test_idx]
        y_train_val, y_test_val = y.to_numpy()[train_idx], y.to_numpy()[test_idx]
        model.fit(X_train_val, y_train_val)
        pred = model.predict(X_test_val)
        scores.append(accuracy_score(y_test_val, pred))
    return scores

class Node:
    def __init__(self, feature=None, threshold=None, true=None, false=None,*,value=None):
        '''
        Class to represent a leaf or a decision node in the decision tree.
        Parameters:
        -----------
        feature: int, default: None
            Feature index that we use to measure the best threshold to get the maximum information gain and decide
            other samples whether it's true (less than or equal with threshold) or false (more than threshold).
        threshold: float, default: None
            The value that used to compare to another samples at the same feature index to determine the decision.
        true: Node, default: None
            Next decision node for the samples where the feature value less than or equal with threshold.
        false: Node, default: None
            Next decision node for the samples where the feature value more than threshold.
        value: float, default: None
            Class prediciton if decision tree, or a float if regression tree.
        '''
        self.feature = feature
        self.threshold = threshold
        self.true = true
        self.false = false
        self.value = value
        
    def is_leaf_node(self):
        return self.value is not None

class DecisionTree:
    '''
    Class too build the decision tree from given data.
    Parameters:
    -----------
    min_samples_split: int, default: 2
        The minimum number of samples to split when building the tree.
    max_depth: int, default: 100
        The maximum depth of the tree.
    n_features: int, deafult: None
        The number of features that the data has.
    '''
    
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None):
        
        self.min_samples_split=min_samples_split
        self.max_depth=max_depth
        self.n_features=n_features
        self.root=None

    def build_tree(self, X, y, depth=0):
        
        ''' 
        A recursive method to build the decision tree by split the data samples with respect to the label on the 
        feature of data and decide the best separates based on impurity.
        '''
        
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))

        # If these stopping criterias are satisfied, then it's a leaf
        if (depth>=self.max_depth or n_labels==1 or n_samples<self.min_samples_split):
            leaf_value = self.calculate_leaf_value(y)
            return Node(value=leaf_value)

        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)

        # find the best split
        best_split = self._best_split(X, y, feat_idxs)

        # create child nodes
        true_idxs, false_idxs = self._split(X[:, best_split['best_feature']], best_split['best_threshold'])
        true = self.build_tree(X[true_idxs, :], y[true_idxs], depth+1)
        false = self.build_tree(X[false_idxs, :], y[false_idxs], depth+1)
        return Node(best_split['best_feature'], best_split['best_threshold'], true, false)

    def _best_split(self, X, y, feat_idxs):
        
        '''
        A function to determine the best split based on feature index, best threshold of the feature and the maximum
        information gain.
        '''
        
        #Dictionary to store the best split data
        best_split = {}
        
        best_gain = -1

        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)

            for thr in thresholds:
                # calculate the information gain
                gain = self._information_gain(y, X_column, thr)

                if gain > best_gain:
                    best_gain = gain
                    best_split['best_feature'] = feat_idx
                    best_split['best_threshold'] = thr
                    best_split['best_gain'] = gain

        return best_split

    def _information_gain(self, y, X_column, threshold):
        
        '''
        A function to get the information gain of the feature index using the gini index of the data.
        '''
        
        # parent gini index
        parent_gini = self.gini_index(y)

        # create children
        true_idxs, false_idxs = self._split(X_column, threshold)

        if len(true_idxs) == 0 or len(false_idxs) == 0:
            return 0
        
        # calculate the weighted avg. gini index of children
        n = len(y)
        n_t, n_f = len(true_idxs), len(false_idxs)
        g_t, g_f = self.gini_index(y[true_idxs]), self.gini_index(y[false_idxs])
        child_gini = (n_t/n) * g_t + (n_f/n) * g_f

        # calculate the IG
        information_gain = parent_gini - child_gini
        return information_gain
    
    def gini_index(self, y):
        
        '''
        A function to calculate the gini index.
        '''
        
        labels = np.unique(y)
        gini = 0
        
        for label in labels:
            point = len(y[y==label])/len(y)
            gini += point**2
        return 1-gini

    def _split(self, X_column, split_thresh):
        
        '''
        A function to split indexes where the given data is less than or more than threshold.
        '''
        
        true_idxs = np.argwhere(X_column <= split_thresh).flatten()
        false_idxs = np.argwhere(X_column > split_thresh).flatten()
        return true_idxs, false_idxs

    def calculate_leaf_value(self, y):
        
        '''
        A function to calculate the leaf value 
        '''
        
        y = list(y)
        return max(y, key=y.count)

    def fit(self, X, y):
        
        '''
        Bulid the decision tree from given data.
        '''
        
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1],self.n_features)
        self.root = self.build_tree(X, y)

    def predict(self, X):
        
        '''
        Classify samples one by one to get a set of labels.
        '''
        return np.array([self.traverse_tree(x, self.root) for x in X])

    def traverse_tree(self, x, node):
       
        '''
        Do a recursive search down the tree and make a prediction the label of the data sampe by the value of the 
        leaf that we end up at.
        '''
        
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self.traverse_tree(x, node.true)
        return self.traverse_tree(x, node.false)
    
    def print_tree(self, tree=None, indent=" "):
        ''' function to print the tree '''
        
        if not tree:
            tree = self.root

        if tree.value is not None:
            print(tree.value)

        else:
            print("X_"+str(tree.feature), "<=", tree.threshold, "?")
            print("%strue:" % (indent), end="")
            self.print_tree(tree.true, indent + indent)
            print("%sfalse:" % (indent), end="")
            self.print_tree(tree.false, indent + indent)

def data_analysis(data):
    flights_data = data
    Flight_data_delay =[]
    for row in flights_data['ARRIVAL_DELAY']:
        if row > 60:
            Flight_data_delay.append(3)
        elif row > 30:
            Flight_data_delay.append(2)
        elif row > 15:
            Flight_data_delay.append(1)
        else:
            Flight_data_delay.append(0)  
    flights_data['Delay'] = Flight_data_delay
    flights_data=flights_data.drop(['YEAR','FLIGHT_NUMBER','AIRLINE','DISTANCE','TAIL_NUMBER','TAXI_OUT','SCHEDULED_TIME','DEPARTURE_TIME','WHEELS_OFF','ELAPSED_TIME','AIR_TIME','WHEELS_ON','DAY_OF_WEEK','TAXI_IN','CANCELLATION_REASON','ORIGIN_AIRPORT', 'DESTINATION_AIRPORT', 'ARRIVAL_TIME', 'ARRIVAL_DELAY', "CANCELLED"],
                                             axis=1)
    flights_data=flights_data.fillna(flights_data.mean())
    flights_random_sampling = flights_data.groupby('Delay').apply(lambda x: x.sample(frac=0.001 if x.Delay.iloc[0]=='0' else 0.01))
    return flights_random_sampling

def main():
    low_memory = False
    flights_data = pd.read_csv('flights.csv')
    flight_data = data_analysis(flights_data)

    X = flight_data.drop(columns=['Delay'])
    y = flight_data['Delay']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    classifier = DecisionTree(max_depth=10)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    print(f'Accuracy: {accuracy_score(y_test, y_pred)*100}%')

    cv = KFold(n_splits=5, random_state = 42, shuffle = True)
    scores = cross_val_score(classifier, X, y, scoring='accuracy', cv=cv)

    print(f'K-Fold Cross Validation Mean Score: {np.mean(scores)}')

if __name__ == "__main__":
    main()