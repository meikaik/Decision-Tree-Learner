import sys, math
from collections import Counter


class Node:
    """
    A node can either be an internal node or a leaf node.
    The val field is used to store either attributes or classifications
    """

    def __init__(self, idx=-1, threshold=0, left=None, right=None, val=""):
        """
        If there is an index and threshold, we know that this is an internal node 
        with an attribute. If not, we know that this is a leaf (classification) node
        
        Internal node: Node(attribute_idx, threshold, >= threshold, < threshold)

        Leaf node: Node(val='true'|'false'|'healthy'|'colic')
        """
        self.idx = idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.val = val
        if idx != -1:
            self.val = attributes[idx]
        if threshold != 0:
            self.val += " " + str(threshold)


def entropy(classifications):
    """
    −SUM(Pi * log2(Pi))
  
    Input: [# of positive, # of negative, etc]
    
    Classifications can be binary (or not). Lower is better
    """
    total = 0
    for classification in classifications:
        p = classification / sum(classifications)
        if p != 0:
            total += p * math.log2(p)
    return -total


def info_gain(prev_classifications, new_classifications):
    """
    Entropy(prev) - SUM(entropy(pi) * pi/p_total)
    
    Higher is better
    """
    remainder = 0
    for classification in new_classifications:
        remainder += (
            entropy(classification) * sum(classification) / sum(prev_classifications)
        )
    return entropy(prev_classifications) - remainder


def classify(data):
    """
    Returns a list of classifications counts
    
    Output: [1 , 3]
    """
    classifications = classify_counter(data)
    return list(classifications.values())


def classify_counter(data):
    """
    Returns a Counter of classifications.
    
    Output: Counter(Female => 1 , Male => 3)
    """
    classifications = Counter()
    for row in data:
        classifications[row[-1]] += 1
    return classifications


def mode(data):
    """
    Returns the most common classification
    """
    counter = classify_counter(data)
    return counter.most_common(1)[0][0]


def partition(data, attribute_idx, threshold):
    """
    Partition the dataset into 2 partitions based on 
    data[row][attribute_idx] >= threshold
    """
    partitions = [], []
    for row in data:
        if row[attribute_idx] >= threshold:
            partitions[0].append(row)
        else:
            partitions[1].append(row)
    return partitions


def choose_attribute(data):
    """
    Chooses the best attribute with the highest information gain in dataset
    """
    best_attribute = (0, 0, 0)
    for attribute_idx in range(len(attributes)):
        # Create a deduped list of all values corresponding to horse_attributes[i]
        values = sorted(list(set([row[attribute_idx] for row in data])))
        # Go throuh each pair in in list of values, partition dataset, and calculate information gain
        for i in range(len(values) - 1):
            # Calculate the midpoint between each pair of values
            threshold = (values[i] + values[i + 1]) / 2
            # Partition dataset based on threshold
            left, right = partition(data, attribute_idx, threshold)
            curr_gain = info_gain(classify(data), [classify(left), classify(right)])
            # If the current information gain is better than the previous best information gain
            best_gain = best_attribute[1]
            if curr_gain > best_gain:
                best_attribute = (attribute_idx, curr_gain, threshold)
    return best_attribute


def id3(data, default):
    """
    Generates a decision tree based on the dataset
    """
    if len(data) == 0:
        return Node(val=default)
    elif len(classify(data)) == 1:
        return Node(val=mode(data))
    attribute_idx, _gain, threshold = choose_attribute(data)
    # print("{} => IG: {} T: {}\n".format(attributes[attribute], _gain, threshold))
    left, right = partition(data, attribute_idx, threshold)
    return Node(attribute_idx, threshold, id3(left, mode(data)), id3(right, mode(data)))


def print_tree(root, space=0, padding=20):
    """
    Source: https://www.geeksforgeeks.org/print-binary-tree-2-dimensions/
    """
    if root is None:
        return
    space += padding
    print_tree(root.right, space)
    print()
    for _ in range(padding, space):
        print(end=" ")
    print(root.val)
    print_tree(root.left, space)


def validate(row, node):
    # Base case: Classification node
    if node.idx is -1:
        return node.val == row[-1]
    # Go left if >= threshold
    if row[node.idx] >= node.threshold:
        return validate(row, node.left)
    # Go right if < threshold
    else:
        return validate(row, node.right)


def parse_data(filename):
    """
    Given filename, return nested list of training data
    
    Input: attr_1, ... , attr_n, classification.
    
    Output: [attr_1, ... , attr_n, classification]
    """
    data = []
    with open(filename) as file:
        for row in file:
            values = []
            cols = row.split(",")
            for col in cols[0:-1]:
                values.append(float(col))
            values.append(cols[-1][:-2])
            data.append(values)
    return data


def print_usage():
    print("Usage: python3 id3.py <train|test>")
    print("For <train>: requires attributes.txt, horseTrain.txt")
    print("For <test>: requires attributes.txt, horseTrain.txt, horseTest.txt")


def train(training_file):
    print("Attributes: {}".format(attributes))
    training_data = parse_data(training_file)
    return id3(training_data, mode(training_data))


def test(test_file, decision_tree):
    test_data = parse_data(test_file)
    correct = 0
    for row in test_data:
        if validate(row, decision_tree):
            correct += 1
    return correct / len(test_data) * 100


def get_attributes(attributes_file):
    global attributes
    with open(attributes_file) as file:
        for row in file:
            attributes = row.split(",")


def main():
    if len(sys.argv) != 2:
        return print_usage()

    if sys.argv[1] == "train":
        get_attributes("attributes.txt")
        tree = train("horseTrain.txt")
        print_tree(tree)
    elif sys.argv[1] == "test":
        get_attributes("attributes.txt")
        tree = train("horseTrain.txt")
        accuracy = test("horseTest.txt", tree)
        print("Accuracy: {}%".format(accuracy))
    else:
        return print_usage()


main()
