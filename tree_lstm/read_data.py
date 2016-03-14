import os
import re
import sys
import codecs

from binary_tree import BinaryTree


def load_ptb_trees(data_dir, dataset="train", binary=False, root_only=True):
    """
    Load the train/dev/test data from the PTB-formatted files
    :param data_dir: data directory
    :param dataset: "train", "dev", or "test"
    :param binary: if True, sentiment values are binary [0/1]; otherwise they are [0/1/2/3/4]
    :param root_only: if True, only return the data for the whole tree (not subtrees)
    :return: dict of the form {index: tree}, where index is (sentence_index).(subtree_index)
    """
    filename = os.path.join(data_dir, dataset + '.txt')

    tree_dict = {}
    vocab = set()

    # read in all the strings, convert them to trees, and store them in a dict
    with codecs.open(filename, 'r', encoding='utf-8') as input_file:
        for line_index, line in enumerate(input_file):
            tree = convert_ptb_to_tree(line)
            seqs_and_masks = tree.get_all_sequences_and_masks(root_only=root_only)
            for node_tuple_index, node_tuple in enumerate(seqs_and_masks):
                key = str(line_index) + '.' + str(node_tuple_index)
                words, left_mask, right_mask, value = node_tuple
                if binary:
                    if value > 2:
                        tree_dict[key] = {'words': words, 'left_mask': left_mask, 'right_mask': right_mask, 'value': 1}
                        vocab.update(set(words))
                    elif value < 2:
                        tree_dict[key] = {'words': words, 'left_mask': left_mask, 'right_mask': right_mask, 'value': 0}
                        vocab.update(set(words))
                else:
                    tree_dict[key] = {'words': words, 'left_mask': left_mask, 'right_mask': right_mask, 'value': value}
                    vocab.update(set(words))

    return tree_dict, vocab


def convert_ptb_to_tree(line):
    index = 0
    tree = None
    line = line.rstrip()
    while '((' in line:
        line = re.sub('\(\(', '( (', line)
    while '))' in line:
        line = re.sub('\)\)', ') )', line)

    stack = []
    parts = line.split()
    for p_i, p in enumerate(parts):
        # opening of a bracket, create a new node, take parent from top of stack
        if p[0] == '(':
            tag = p[1:]
            if tree is None:
                tree = BinaryTree(index, tag)
            else:
                add_descendant(tree, index, tag, stack[-1])
            # add the newly created node to the stack and increment the index
            stack.append(index)
            index += 1
        # otherwise, update the word of the node on top of the stack, and pop it
        elif p[-1] == ')':
            tag = p[:-1]
            if tag != '':
                tree.set_word(index-1, tag)
            stack.pop(-1)
        else:
            # deal with a couple of edge cases
            parts[p_i+1] = p + '_' + parts[p_i+1]
    return tree


def add_descendant(tree, index, tag, parent_index):
    # add to the left first if possible, then to the right
    if tree.has_left_descendant_at_node(parent_index):
        if tree.has_right_descendant_at_node(parent_index):
            sys.exit("Node " + str(parent_index) + " already has two children")
        else:
            tree.add_right_descendant(index, tag, parent_index)
    else:
        tree.add_left_descendant(index, tag, parent_index)


