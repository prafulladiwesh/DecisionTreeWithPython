import numpy as np
import pandas as pd
import math
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--data")
parser.add_argument("--output")
args = parser.parse_args()
print("data: {}".format(args.data))
print("output: {}".format(args.output))

# Parsed input data
input_file = args.data
output_file = args.output
data_frame = pd.read_csv(input_file, header=None)


def calculate_entropy(data):
    row_count = 0
    array_row = []
    group_class = data.groupby(group_data_by_last_column)
    row_count = create_data_group(array_row, group_class, row_count)
    return np.sum([abs((array_row[row]/row_count) * (math.log(array_row[row] / row_count, len(class_label_count))))
                   for row in range(len(array_row))])


def create_data_group(array_row, group_class, row_count):
    for grouped_value, grouped_data in group_class:
        grouped_row, grouped_column = grouped_data.shape
        array_row.append(grouped_row)
        # total number of rows of grouped class labels
        row_count = row_count + grouped_row
    return row_count


def calculate_information_gain(data, main_entropy):
    information_gain_list = []
    for header in header_data:
        group = data.groupby(data[data.columns[header]])
        split_fraction_of_attributes = []
        entropy_of_attribute = []
        split_attribute_entropy(entropy_of_attribute, group, split_fraction_of_attributes)
        calculate_weighted_info_gain(entropy_of_attribute, information_gain_list, main_entropy,
                                     split_fraction_of_attributes)
    return information_gain_list.index(max(information_gain_list))


def calculate_weighted_info_gain(entropy_of_attribute, information_gain_list, main_entropy,
                                 split_fraction_of_attributes):
    # entropy of each split of the branch
    weighted_entropy = np.sum([(split_fraction_of_attributes[split] * entropy_of_attribute[split])
                               for split in range(len(split_fraction_of_attributes))])
    # add to the list of all information gains of all the attributes.
    # Max information gain will be selected as split attribute
    information_gain_list.append(np.subtract(main_entropy, weighted_entropy))


def split_attribute_entropy(entropy_of_attribute, group, split_fraction_of_attributes):
    for attribute, data_group in group:
        grouped_class = data_group.groupby(data_group.columns[group_data_by_last_column])
        attribute_count = 0
        previous_attribute = attribute
        attribute_count = create_branch(attribute, attribute_count, grouped_class, previous_attribute)
        entropy = calculate_entropy(data_group)
        entropy_of_attribute.append(entropy)
        # calculate fraction of data in each branch split
        split_fraction_of_attributes.append(np.divide(attribute_count, data_frame_rows))


def create_branch(attribute, attribute_count, grouped_class, previous_attribute):
    for grouped_value, grouped_data in grouped_class:
        grouped_rows, grouped_columns = grouped_data.shape
        if previous_attribute == attribute:
            attribute_count = attribute_count + grouped_rows
    return attribute_count


def build_decision_tree(data, node_entropy_str, feature_str, value_str, node_str, bracket_str):
    split_node = calculate_information_gain(data, calculate_entropy(data))
    split_by_root_node = data.groupby(split_node, sort=False)
    for grouped_value, grouped_data in split_by_root_node:
        # calculate entropy for each node.
        entropy = calculate_entropy(grouped_data)
        file_op.write(node_entropy_str + str(entropy) + feature_str + str(split_node) + value_str +
                      str(grouped_value) + bracket_str)
        # recursive call to decision tree if the entropy is not zero.
        # create subtree if the entropy is zero.
        build_decision_tree(grouped_data, node_entropy_str, feature_str, value_str, node_str, bracket_str) \
            if entropy != 0.00 else file_op.write(str(grouped_data[group_data_by_last_column].unique()[0]))
        file_op.write(node_str)


def run(data):
    tree_entropy_str = '<tree entropy="'
    node_entropy_str = '<node entropy="'
    feature_str = '" feature="att'
    value_str = '" value="'
    node_str = "</node>"
    tree_str = '</tree>'
    bracket_str = '">'
    entropy_tree = calculate_entropy(data)
    file_op.write(tree_entropy_str + str(entropy_tree) + bracket_str)
    # call to build decision tree
    build_decision_tree(data, node_entropy_str, feature_str, value_str, node_str, bracket_str)
    file_op.write(tree_str)


data_frame_rows, data_frame_columns = data_frame.shape
group_data_by_last_column = data_frame_columns - 1
# frequency counts of last column (class labels)
class_label_count = data_frame[group_data_by_last_column].value_counts()
header_data = list(data_frame.columns.values)
# remove last column (class labels) from the data
header_data.remove(group_data_by_last_column)
file_op = open(output_file, 'w')
run(data_frame)
