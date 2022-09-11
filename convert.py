import torch
import info_util
import numpy as np
import os
import lxml.etree as et
import argparse
import textwrap


def make_directory(base_path: str) -> int:
    """
        Checks if a directory exists and if doesn't creates the directory.
        Args:
        base_path : Directory path which will be created if it doesn't exist.
        Returns 0 if directory exists else 1
    """
    if os.path.exists(base_path):
        return 0

    # Create the directory since the path doesn't exist.
    os.makedirs(base_path)
    if os.path.exists(base_path):
        return 0

    # Path doesn't exist as well as directory couldn't be created.
    print("Error : Cannot create desired path : ", base_path)
    return 1


def generate_csv(csv_name: str, weight_matrix: torch.tensor, base_path: str, transpose=False) -> str:
    """
        Generates csv for weights or bias matrix.
        Args:
        csv_name : A string name for csv file which will store the weights.
        weight_matrix : A torch tensor holding weights that will be stored in the matrix.
        base_path : Base path where csv will be stored.
    """
    # Check if base path exists else create directory.
    make_directory(base_path)
    file_path = os.path.join(base_path, csv_name)
    matrix = weight_matrix.numpy().ravel()
    np.savetxt(file_path, matrix, fmt='%1.128f')
    if transpose:
        matrix = weight_matrix.numpy().transpose().ravel()
        np.savetxt(file_path, matrix, fmt='%1.128f')
        print("Transposed")
    return file_path


def param_dictionary(layer_info, base_path):
    """
        Generates info for layer's weights or bias matrix.
        Args:
        layer_info : Torchinfo Object where layer info is stored.
        base_path : Base path where csv will be stored.
    """
    p_dict = {}
    p_dict["is_leaf"] = 1
    p_dict["depth"] = layer_info.depth
    p_dict["var_name"] = layer_info.var_name
    p_dict["trainable_param"] = layer_info.leftover_params()
    if (hasattr(layer_info.module, 'weight') and layer_info.module.weight != None):
        p_dict["has_weight"] = 1
        w_csv = 'weight_' + str(layer_info.layer_id) + '.csv'
        p_dict["weight_csv"] = generate_csv(
            w_csv, layer_info.module.weight.detach(), base_path)
    else:
        p_dict["has_weight"] = 0
        p_dict["weight_offset"] = 0
        p_dict["weight_csv"] = "None"
    if (hasattr(layer_info.module, 'bias') and layer_info.module.bias != None):
        p_dict["has_bias"] = 1
        b_csv = 'bias_' + str(layer_info.layer_id) + '.csv'
        p_dict["bias_csv"] = generate_csv(
            b_csv, layer_info.module.bias.detach(), base_path)
    else:
        p_dict["has_bias"] = 0
        p_dict["bias_csv"] = "None"
    for k, v in layer_info.module.state_dict().items():
        if k in ('weight', 'bias'):
            continue
        p_dict['has_' + k] = 1
        csv_name = k + '_' + str(layer_info.layer_id) + '.csv'
        p_dict[k + "_csv"] = generate_csv(csv_name,
                                          layer_info.module.state_dict()[k].detach(), base_path)
    return p_dict


def generate_xml(model, output_path):
    """
        Generates xml for given PyTorch model.
        Args:
        model : PyTorch model which you want to parse.
        output_path : Base path where output files will be stored.
    """
    temp = info_util.summary_gg(model, (1, 3, 224, 224))
    root = et.Element(temp[0].class_name)
    child = et.SubElement(root, "trainable_param")
    child.text = str(temp[0].num_params)
    base = temp[0].class_name + '_' + str(temp[0].layer_id)
    make_directory(base)
    tree = et.ElementTree(root)
    parent = {0: [root, base]}
    for layer_info in temp[1:]:
        level = layer_info.depth
        class_name = layer_info.class_name
        base = parent[level - 1][1]
        # create the element and link it to its parent
        elem = et.SubElement(parent[level - 1][0],
                             "layer", {'name': class_name})
        parameter_dictionary = {}
        parameter_dictionary["var_name"] = layer_info.var_name
        parameter_dictionary["is_leaf"] = 0
        if (layer_info.is_leaf_layer):
            parameter_dictionary = param_dictionary(
                layer_info, os.path.join(base, layer_info.var_name))
        # create children to hold the other data items
        for k, v in parameter_dictionary.items():
            child = et.SubElement(elem, k)
            child.text = str(v)
        # record current element as a possible parent
        parent[level] = [elem, os.path.join(base, layer_info.var_name)]

    with open(output_path, 'wb') as f:
        tree.write(f, pretty_print=True)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        prog='Download and parse model script (PyTorch -> Mlpack)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent('''\
        Model weight conversion script!
        --------------------------------
        This script is used to download the model weight and convert it to csv.
        Also generates the xml file which contains the information about the 
        model.
        Usage: --model model_name
        --------------------------------
        '''))

    parser.add_argument('--model', metavar="model name",
                        type=str, help="Enter model name to process", required=True)
    args = parser.parse_args()
    model = torch.hub.load(
        'shubham1206agra/pretrained-models.pytorch', args.model)

    generate_xml(model, args.model + '.xml')
