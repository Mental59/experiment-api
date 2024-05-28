import json
import uuid
import os
import re

import networkx as nx
from fastapi import UploadFile, HTTPException

from app.constants.resources import ONTO_PATH

from ...models.custom_parser.python_parser import FuncCall
from ...services.custom_parser.python_parser import parse as parse_python_code
from ...core.exceptions import create_exception_details


class OntoParser:
    init_node_name = "init"

    def __init__(self, data):
        self.raw_data = data
        self.last_id = int(data["last_id"])
        self.namespaces = data["namespaces"]

        self.nodes = {node["id"]: node for node in data["nodes"]}

        self.relations: list = data["relations"]
        self.hashed_relations = {f"{relation['source_node_id']}-{relation['destination_node_id']}": relation for relation in data["relations"]}

        self.digraph = self.create_digraph()

    def create_digraph(self):
        init_node = self.get_node_by_name("init")

        graph = nx.DiGraph()

        for node_id in filter(lambda node_id: node_id != init_node["id"], self.nodes):
            graph.add_node(node_id)

        for relation in filter(lambda relation: relation["source_node_id"] != init_node["id"] and relation["destination_node_id"] != init_node["id"], self.relations):
            graph.add_edge(relation["source_node_id"], relation["destination_node_id"])
            
        return graph
    
    def get_link_between(self, _from, _to):
        return self.hashed_relations.get(f"{_from['id']}-{_to['id']}")
    
    def shortest_path(self, _from, _to, undirected=True):
        try:
            shortest_path = nx.shortest_path(nx.to_undirected(self.digraph) if undirected else self.digraph, _from["id"], _to["id"])
            shortest_path = [self.get_node_by_id(node_id) for node_id in shortest_path]
            return shortest_path
        except Exception:
            return []

    @classmethod
    def load_from_file(cls, path, encoding="utf-8"):
        with open(path, encoding=encoding) as f:
            data = json.load(f)

        cls.__validate_data(data)
        
        return cls(data)
    
    @classmethod
    def load_from_text(cls, text):
        data = json.loads(text)

        cls.__validate_data(data)
        
        return cls(data)
    
    def get_node_by_id(self, node_id: str):
        return self.nodes[node_id]
    
    def get_node_by_name(self, node_name: str):
        for node in self.nodes.values():
            if node["name"] == node_name:
                return node
    
    def get_nodes_linked_to(self, node, link_names = None):
        """ get all nodes connected to node with node_id by links with link_names"""
        node_id = node["id"]

        nodes = [
            self.nodes[relation["source_node_id"]] for relation in self.relations if (
                relation["destination_node_id"] == node_id and (link_names is None or relation["name"] in link_names)
            )
        ]

        return nodes
    
    def get_nodes_linked_from(self, node, link_names = None):
        """get all nodes connected from node with node_id by links with link_names"""
        node_id = node["id"]

        nodes = [
            self.nodes[relation["destination_node_id"]] for relation in self.relations if (
                relation["source_node_id"] == node_id and (link_names is None or relation["name"] in link_names)
            )
        ]

        return nodes
    
    def add_node(self, name: str, attributes: dict | None = None):
        if attributes is None:
            attributes = dict()

        node = dict(attributes=attributes, id=str(self.last_id), name=name, namespace=self.namespaces["default"], position_x=0.0, position_y=0.0)
        self.digraph.add_node(node["id"])
        self.last_id += 1

        self.nodes[node["id"]] = node

        return node
    
    def add_relation(self, name: str, source_node_id: str, destination_node_id: str, attributes: dict | None = None):
        if attributes is None:
            attributes = dict()
        
        relation = dict(attributes=attributes, destination_node_id=destination_node_id, id=str(self.last_id), name=name, namespace=self.namespaces["default"], source_node_id=source_node_id)
        self.digraph.add_edge(source_node_id, destination_node_id)
        self.last_id += 1

        self.relations.append(relation)
        self.hashed_relations[f"{relation['source_node_id']}-{relation['destination_node_id']}"] = relation

        return relation

    def add_experiment(self, name: str, attributes: dict | None = None, base_experiment_id: str | None = None):
        experiment_node = self.get_node_by_name("Experiment")

        experiments = self.get_nodes_linked_to(experiment_node, ["is_a"])
        base_experiment = next((exp for exp in experiments if exp["attributes"]["tracker_info"]["run_id"] == base_experiment_id), None)

        node = self.add_node(name, {**attributes, **dict(branch="Experiment", leaf=True)})
        self.add_relation("is_a", node["id"], experiment_node["id"])

        if base_experiment is not None:
            self.add_relation("based_on", node["id"], base_experiment["id"])
        
        self.save(ONTO_PATH)

    def save(self, path: str):
        self.raw_data["last_id"] = str(self.last_id)
        self.raw_data["nodes"] = list(self.nodes.values())
        self.raw_data["relations"] = self.relations

        with open(path, "w", encoding="utf-8") as file:
            json.dump(self.raw_data, file, indent=4)

    def get_main_branch_nodes(self):
        return self.get_nodes_linked_from(self.get_node_by_name(self.init_node_name))
    
    def get_leaf_nodes_for_branch(self, branch_name: str):
        leaves = [node for node in self.nodes.values() if node["attributes"].get("leaf", False) and node["attributes"].get("branch") == branch_name]
        return leaves
    
    def get_main_branches_tree_view(self):
        main_branche_nodes = self.get_main_branch_nodes()

        node_paths = []
        for branch_node in main_branche_nodes:
            leaf_nodes = self.get_leaf_nodes_for_branch(branch_node["name"])

            shortest_paths = []
            for leaf_node in leaf_nodes:
                shortest_path = self.shortest_path(branch_node, leaf_node)
                shortest_paths.append(shortest_path)

                for combination_node in self.get_nodes_linked_from(leaf_node, ["combination"]):
                    shortest_paths.append(shortest_path[0:-1] + [combination_node])
            
            if len(leaf_nodes) == 0:
                shortest_paths.append([branch_node])

            node_paths.extend(shortest_paths)

        tree = dict()
        for node_path in [node_path for node_path in node_paths if len(node_path) > 0]:
            if node_path[0]["id"] not in tree:
                tree[node_path[0]["id"]] = dict(data=node_path[0], children=dict())
            current_node = tree[node_path[0]["id"]]

            for node in node_path[1:]:
                if node["id"] not in current_node["children"]:
                    current_node["children"][node["id"]] = dict(data=node, children=dict())
                current_node = current_node["children"][node["id"]]
        
        self.__tree_dicts_to_tree_lists(tree)

        return list(tree.values())
    
    def find_func_calls(self, func_calls: list[FuncCall]):
        model_nodes = self.get_nodes_linked_to(self.get_node_by_name("Machine Learning Method"), ["is_a"])

        res = []
        for model_node in model_nodes:
            model_lib_nodes = self.get_nodes_linked_to(model_node, ["use"])
            parameter_nodes = self.get_nodes_linked_from(model_node, ["takes_parameter"])
            for model_lib_node in model_lib_nodes:
                func_calls_for_model = [fc for fc in func_calls if fc.get_full_name() == model_lib_node["name"]]

                if len(func_calls_for_model) == 0:
                    continue
                
                if len(parameter_nodes) > 0:
                    parameter_node = parameter_nodes[0]
                    attrs = parameter_node["attributes"]
                    for func_call in func_calls_for_model:
                        func_call_arg = func_call.get_arg_by_keyword_or_index(index=attrs["index"], keyword=attrs["keyword"])
                        if func_call_arg.value is not None and attrs["value"] in str(func_call_arg.value):
                            res.append(model_lib_node)
                            break
                else:
                    res.append(model_lib_node)

        return res
    
    def get_model_nodes(self):
        model_nodes = self.get_nodes_linked_to(self.get_node_by_name("Machine Learning Method"), ["is_a"])
        for model_node in model_nodes:
            combination_nodes = self.get_nodes_linked_from(model_node, ['combination'])
            model_nodes.extend(combination_nodes)
        return model_nodes
    
    def get_ml_tasks(self) -> list[dict]:
        machine_learning_node = self.get_node_by_name('Machine Learning')
        leaf_ml_task_nodes = self.get_leaf_nodes_for_branch('Task')
        result = []
        for leaf in leaf_ml_task_nodes:
            path = self.shortest_path(machine_learning_node, leaf, undirected=True)
            result.append([p for p in path[1:]])
        return result
    
    def add_ml_task(self, new_node_name: str, parent_node_id: str | None = None):
        machine_learning_node = self.get_node_by_name('Machine Learning')

        node = self.add_node(new_node_name, attributes={"branch": "Task", "leaf": True})
        if parent_node_id is None:
            self.add_relation('is_a', node['id'], machine_learning_node['id'])
        else:
            parent_node = self.nodes.get(parent_node_id)
            if parent_node is None:
                raise HTTPException(status_code=400, detail=create_exception_details('Родительский узел онтологии не найден'))
            if parent_node['attributes'].get('leaf', False):
                parent_node['attributes'] = dict()
            self.add_relation('is_a', node['id'], parent_node['id'])

        self.save(ONTO_PATH)
    
    def add_ml_transformer_model(self, parent_node_id: str, node_name: str, model_name_or_path: str):
        parent_node = self.nodes.get(parent_node_id)
        if parent_node is None:
            raise HTTPException(status_code=400, detail=create_exception_details('Родительский узел онтологии не найден'))
        node = self.add_node(
            node_name,
            attributes=dict(
                branch="Method",
                leaf=True,
                id=str(uuid.uuid4()),
                model_name_or_path=model_name_or_path,
                transformer=True
            )
        )
        self.add_relation('used_for', node['id'], parent_node['id'])

        transformer_method = self.get_node_by_name('transformers.AutoModelForTokenClassification.from_pretrained')
        self.add_relation('use', transformer_method['id'], node['id'])

        parameter_node = self.add_node(
            '.',
            attributes={
                "index": 0,
                "keyword": "pretrained_model_name_or_path",
                "value": model_name_or_path
            }
        )
        self.add_relation('takes_parameter', node['id'], parameter_node['id'])

        method_node = self.get_node_by_name('Machine Learning Method')
        self.add_relation('is_a', node['id'], method_node['id'])

        self.save(ONTO_PATH)

    @staticmethod
    def __tree_dicts_to_tree_lists(tree: dict):
        if len(tree) == 0:
            return

        for key in tree:
            OntoParser.__tree_dicts_to_tree_lists(tree[key]["children"])
            tree[key]["children"] = list(tree[key]["children"].values())
    
    @staticmethod
    def __validate_data(data):
        if not data:
            raise ValueError("corrupt ontology")
        
        if not ("nodes" in data):
            raise ValueError("corrupt ontology, nodes missing")
        
        if not ("relations" in data):
            raise ValueError("corrupt ontology, relations missing")


async def extract_func_calls(source_files: list[UploadFile]) -> list[FuncCall]:
    supported_extensions = ['.py', '.ipynb']

    extensions = [os.path.splitext(source_file.filename)[-1] for source_file in source_files]
    for extension in extensions:
        if extension not in supported_extensions:
            raise HTTPException(status_code=400, detail=create_exception_details(f'Unsupported file extension {extension}'))

    res = []
    for source_file, extension in zip(source_files, extensions):
        try:
            content_bytes = await source_file.read()
            text = content_bytes.decode("utf-8")
        except ValueError as error:
            raise HTTPException(400, detail=create_exception_details(f"Invalid source file; reason: {error}"))
        
        func_calls = []
        if extension == '.py':
            func_calls = parse_python_code(text)
        elif extension == '.ipynb':
            code_lines = []
            data = json.loads(text)
            for cell in data['cells']:
                if cell['cell_type'] == 'code':
                    code_lines.extend([line for line in cell['source'] if re.match(r"%%[a-zA-Z]*", line) is None] + ['\n'])
            text = ''.join(code_lines)
            func_calls = parse_python_code(text)

        res.extend(func_calls)
    return res


async def find_models(onto: OntoParser, source_files: list[UploadFile]):
    ids = set()
    res_models = []

    for source_file in source_files:
        onto_func_calls = onto.find_func_calls(await extract_func_calls(source_files=[source_file]))
        models = [onto.get_nodes_linked_from(func_call, ["use"])[0] for func_call in onto_func_calls]

        # find model combinations
        model_combinations = []
        excluded_model_ids = set() # exclude models that are used as a combination of models (for example, lstm and crf are used as one model lstm-crf)
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                path_i_j = onto.shortest_path(models[i], models[j], undirected=False)
                path_j_i = onto.shortest_path(models[j], models[i], undirected=False)
                path = None

                if len(path_i_j) == 3:
                    path = path_i_j  
                if len(path_j_i) == 3:
                    path = path_j_i

                if path is not None:
                    link_1 = onto.get_link_between(path[0], path[1])
                    link_2 = onto.get_link_between(path[1], path[2])
                    if link_1["name"] == "combination" and link_2["name"] == "with":
                        model_combinations.append(path[1])
                        excluded_model_ids.update([models[i]["id"], models[j]["id"]])

        models.extend(model_combinations)
        models = [model for model in models if model["id"] not in excluded_model_ids and model["id"] not in ids]
        ids.update([model["id"] for model in models])

        libraries = [dict(id=lib["id"], name=lib["name"]) for lib in onto_func_calls]
        res_models.extend([dict(id=model["id"], name=model["name"], libraries=libraries) for model in models])

    return res_models


def main():
    onto_parser = OntoParser.load_from_file(r"G:\PythonProjects\ExperimentApp\ExperimentApi\resources\wine-recognition.ont")
    main_branches = onto_parser.get_main_branches_tree_view()


if __name__ == '__main__':
    main()
