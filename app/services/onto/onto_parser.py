import json

from ...models.parser.python_parser import FuncCall


class OntoParser:
    def __init__(self, data):
        self.last_id = data["last_id"]
        self.namespaces = data["namespaces"]
        self.nodes = {node["id"]: node for node in data["nodes"]}
        self.relations = data["relations"]

        self.model_nodes = []
        for model_node in self.get_nodes_linked_to(self.get_node_by_name("Machine Learning Method"), ["is_a"]):
            self.model_nodes.extend(self.get_nodes_linked_to(model_node, ["use"]))

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
        """ get all nodes connected to node with node_id by link with link_name"""
        node_id = node["id"]

        nodes = [
            self.nodes[relation["source_node_id"]] for relation in self.relations if (
                relation["destination_node_id"] == node_id and (link_names is None or relation["name"] in link_names)
            )
        ]

        return nodes
    
    def get_nodes_linked_from(self, node, link_names = None):
        """get all nodes connected from node with node_id by link with link_name"""
        node_id = node["id"]

        nodes = [
            self.nodes[relation["destination_node_id"]] for relation in self.relations if (
                relation["source_node_id"] == node_id and (link_names is None or relation["name"] in link_names)
            )
        ]

        return nodes
    
    def find_func_calls(self, func_calls: list[FuncCall]):
        func_call_full_names = set([func_call.get_full_name() for func_call in func_calls])
        return [model_node for model_node in self.model_nodes if model_node["name"] in func_call_full_names]
    
    def get_models_tree_view(self):
        ml_node = self.get_node_by_name("Machine Learning Method")

        node_paths = []
        for model_node in self.get_nodes_linked_to(ml_node, ["is_a"]):
            nodes = [model_node]
            task_nodes = self.get_nodes_linked_from(model_node, ["used_for"])
            
            if len(task_nodes) > 0:
                task_node = task_nodes[0]

                while task_node["name"] != "Task":
                    nodes.append(task_node)
                    task_nodes = self.get_nodes_linked_from(task_node, ["is_a"])
                    if len(task_nodes) == 0:
                        break
                    else:
                        task_node = task_nodes[0]

            node_paths.append(nodes[::-1])

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
