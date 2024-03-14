import json

import networkx as nx
from fastapi import UploadFile, HTTPException

from ...models.parser.python_parser import FuncCall
from ...services.parser.python_parser import parse as parse_python_code
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
        graph = nx.DiGraph()
        for node_id in self.nodes:
            graph.add_node(node_id)
        for link in self.relations:
            graph.add_edge(link["source_node_id"], link["destination_node_id"])
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

    def add_experiment(self, name: str, attributes: dict | None = None):
        experiment_node = self.get_node_by_name("Experiment")
        node = self.add_node(name, attributes)
        self.add_relation("a_part_of", experiment_node["id"], node["id"])

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
        model_nodes = []
        for model_node in self.get_nodes_linked_to(self.get_node_by_name("Machine Learning Method"), ["is_a"]):
            model_nodes.extend(self.get_nodes_linked_to(model_node, ["use"]))
        
        func_call_full_names = set([func_call.get_full_name() for func_call in func_calls])
        return [model_node for model_node in model_nodes if model_node["name"] in func_call_full_names]
    
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


async def find_models(onto: OntoParser, source_files: list[UploadFile]):
    ids = set()
    res_models = []

    for source_file in source_files:
        try:
            content_bytes = await source_file.read()
            text = content_bytes.decode("utf-8")
        except ValueError as error:
            raise HTTPException(400, detail=create_exception_details(f"Invalid source file; reason: {error}"))
        
        func_calls = onto.find_func_calls(parse_python_code(text))
        models = [onto.get_nodes_linked_from(func_call, ["use"])[0] for func_call in func_calls]


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

        libraries = [dict(id=lib["id"], name=lib["name"]) for lib in func_calls]
        res_models.extend([dict(id=model["id"], name=model["name"], libraries=libraries) for model in models])

    return res_models


def main():
    onto_parser = OntoParser.load_from_file(r"G:\PythonProjects\ExperimentApp\ExperimentApi\resources\wine-recognition.ont")
    main_branches = onto_parser.get_main_branches_tree_view()


if __name__ == '__main__':
    main()
