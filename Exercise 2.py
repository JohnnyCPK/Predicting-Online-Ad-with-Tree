# Importing the required libraries
import matplotlib.pyplot as plt  # For creating visualizations
import numpy as np  # For numerical operations

# Generate an array of 1000 values ranging from 0 to 1 (inclusive) to represent positive fractions
pos_fraction = np.linspace(0.00, 1.00, 1000)

# Calculate the Gini impurity for each value of positive fraction
# Gini impurity formula: Gini = 1 - (p^2 + (1-p)^2)
# where p is the positive fraction, and (1-p) is the negative fraction
# Lower Gini is better
gini = 1 - pos_fraction**2 - (1 - pos_fraction)**2

# Plot the Gini impurity against the positive fraction
plt.plot(pos_fraction, gini)

# Set the y-axis limits to range from 0 to 1 (Gini impurity values are between 0 and 1)
plt.ylim(0, 1)

# Label the x-axis to indicate it represents the positive fraction
plt.xlabel('Positive Fraction')

# Label the y-axis to indicate it represents the Gini Impurity
plt.ylabel('Gini Impurity')

# Display the plot
plt.show()

# In binary cases, if the positive fraction is 50%, the highest impurity is 0.5; 
# If it's 100% or 0%, the impurity is 0

# Define a function to calculate Gini impurity for a given set of labels
def gini_impurity(labels):
    # When set is empty, it is also pure
    if len(labels) == 0:
        return 0
    # Count the occurrence of each label
    counts = np.unique(labels, return_counts=True)[1]
    fractions = counts / float(len(labels))
    return 1 - np.sum(fractions ** 2)

# Test the Gini impurity function with different sets of labels
print(f"Gini Impurity (Example 1): {gini_impurity([1,1,0,1,0]):.4f}")
print(f"Gini Impurity (Example 2): {gini_impurity([1,1,0,1,0,0]):.4f}")
print(f"Gini Impurity (Example 3): {gini_impurity([1,1,1,1]):.4f}")

# Generate positive fractions for entropy calculation
pos_fraction = np.linspace(0.001, 0.999, 1000)

# Calculate entropy for each value of positive fraction
ent = - (pos_fraction * np.log2(pos_fraction) + (1 - pos_fraction) * np.log2(1 - pos_fraction))

# Plot the entropy against the positive fraction
plt.plot(pos_fraction, ent)
plt.xlabel('Positive Fraction')
plt.ylabel('Entropy')
plt.ylim(0, 1)
plt.show()

# From this graph, if the positive fraction is 50%, the entropy is at its highest at 1; 
# If it's 100% or 0%, the entropy is 0

# Define a function to calculate entropy for a given set of labels
def entropy(labels):
    if len(labels) == 0:
        return 0
    counts = np.unique(labels, return_counts=True)[1]
    fractions = counts / float(len(labels))
    return - np.sum(fractions * np.log2(fractions))

# Test the entropy function with different sets of labels
print(f"Entropy (Example 1): {entropy([1,1,0,1,0]):.4f}")
print(f"Entropy (Example 2): {entropy([1,1,0,1,0,0]):.4f}")
print(f"Entropy (Example 3): {entropy([1,1,1,1]):.4f}")

# Dictionary to store impurity calculation functions
criterion_function = {'gini': gini_impurity,
                      'entropy': entropy}

# Define a function to calculate the weighted impurity of children after a split
def weighted_impurity(groups, criterion='gini'):
    """
    Calculate weighted impurity of children after a split.
    @param groups: list of children, and a child consists of a list of class labels.
    @param criterion: metrics to measure the quality of a split,
                      'gini' for Gini Impurity or 'entropy' for Information Gain.
    """
    total = sum(len(group) for group in groups)
    weighted_sum = 0.0
    for group in groups:
        weighted_sum += len(group) / float(total) * criterion_function[criterion](group)
    return weighted_sum

# Example splits and their weighted impurities
children_1 = [[1, 0, 1], [0, 1]]
children_2 = [[1, 1], [0, 0, 1]]

print(f"Weighted Entropy (Split 1): {weighted_impurity(children_1, 'entropy'):.4f}")
print(f"Weighted Entropy (Split 2): {weighted_impurity(children_2, 'entropy'):.4f}")

# Define a function to split a node based on a feature and value
def split_node(X, y, index, value):
    x_index = X[:, index]
    if X[0, index].dtype.kind in ['i', 'f']:
        mask = x_index >= value
    else:
        mask = x_index == value
    left = [X[~mask, :], y[~mask]]
    right = [X[mask, :], y[mask]]
    return left, right

# Define a function to find the best split for a dataset
def get_best_split(X, y, criterion):
    best_index, best_value, best_score, children = None, None, 1, None
    for index in range(len(X[0])):
        for value in np.sort(np.unique(X[:, index])):
            groups = split_node(X, y, index, value)
            impurity = weighted_impurity([groups[0][1], groups[1][1]], criterion)
            if impurity < best_score:
                best_index, best_value, best_score, children = index, value, impurity, groups
    return {'index': best_index, 'value': best_value, 'children': children}

# Define a function to determine the leaf value (majority label)
def get_leaf(labels):
    return np.bincount(labels).argmax()

# Define a function to recursively split a tree node
def split(node, max_depth, min_size, depth, criterion):
    left, right = node['children']
    del node['children']
    if left[1].size == 0:
        node['right'] = get_leaf(right[1])
        return
    if right[1].size == 0:
        node['left'] = get_leaf(left[1])
        return
    if depth >= max_depth:
        node['left'], node['right'] = get_leaf(left[1]), get_leaf(right[1])
        return
    if left[1].size <= min_size:
        node['left'] = get_leaf(left[1])
    else:
        result = get_best_split(left[0], left[1], criterion)
        node['left'] = result
        split(node['left'], max_depth, min_size, depth + 1, criterion)
    if right[1].size <= min_size:
        node['right'] = get_leaf(right[1])
    else:
        result = get_best_split(right[0], right[1], criterion)
        node['right'] = result
        split(node['right'], max_depth, min_size, depth + 1, criterion)

# Define a function to train a decision tree
def train_tree(X_train, y_train, max_depth, min_size, criterion='gini'):
    X = np.array(X_train)
    y = np.array(y_train)
    root = get_best_split(X, y, criterion)
    split(root, max_depth, min_size, 1, criterion)
    return root

# Training data and labels
X_train = [['tech', 'professional'],
           ['fashion', 'student'],
           ['fashion', 'professional'],
           ['sports', 'student'],
           ['tech', 'student'],
           ['tech', 'retired'],
           ['sports', 'professional']]

y_train = [1, 0, 0, 0, 1, 0, 1]

# Train the tree
tree = train_tree(X_train, y_train, 2, 2)

# Define conditions for visualization
CONDITION = {'numerical': {'yes': '>=', 'no': '<'},
             'categorical': {'yes': 'is', 'no': 'is not'}}

# Define a function to visualize the tree
def visualize_tree(node, depth=0):
    if isinstance(node, dict):
        if isinstance(node['value'], (int, float)):
            condition = CONDITION['numerical']
        else:
            condition = CONDITION['categorical']
        print(f"{' ' * depth * 3}If X{node['index'] + 1} {condition['no']} {node['value']}:")
        if 'left' in node:
            visualize_tree(node['left'], depth + 1)
        print(f"{' ' * depth * 3}Else If X{node['index'] + 1} {condition['yes']} {node['value']}:")
        if 'right' in node:
            visualize_tree(node['right'], depth + 1)
    else:
        print(f"{' ' * depth * 3}Class: {node}")

# Visualize the tree
visualize_tree(tree)

# Numeric dataset for training
X_train_n = [[6, 7],
             [2, 4],
             [7, 2],
             [3, 6],
             [4, 7],
             [5, 2],
             [1, 6],
             [2, 0],
             [6, 3],
             [4, 1]]

y_train_n = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]

# Train the tree on numeric data
tree = train_tree(X_train_n, y_train_n, 2, 2)

# Visualize the tree for numeric data
visualize_tree(tree)

# Use sklearn's DecisionTreeClassifier for comparison
from sklearn.tree import DecisionTreeClassifier
tree_sk = DecisionTreeClassifier(criterion='gini', max_depth=2, min_samples_split=2)
tree_sk.fit(X_train_n, y_train_n)

# Export the sklearn tree as a DOT file for visualization
from sklearn.tree import export_graphviz
export_graphviz(tree_sk, out_file='tree.dot',
                feature_names=['X1', 'X2'], impurity=False,
                filled=True, class_names=['0', '1'])
#TODO: Graphviz Installation to get dot file