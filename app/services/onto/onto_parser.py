import json

from ...models.parser.python_parser import FuncCall


class OntoParser:
    def __init__(self, data):
        self.last_id = data["last_id"]
        self.namespaces = data["namespaces"]
        self.nodes = {node["id"]: node for node in data["nodes"]}
        self.relations = data["relations"]

        self.model_nodes = []
        for model_node in self.get_nodes_linked_to(self.get_node_by_name("Machine Learning Method"), "is_a"):
            self.model_nodes.extend(self.get_nodes_linked_to(model_node, "use"))

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
    
    def get_nodes_linked_to(self, node, link_name = None):
        """ get all nodes connected to node with node_id by link with link_name"""
        node_id = node["id"]

        nodes = [
            self.nodes[relation["source_node_id"]] for relation in self.relations if (
                relation["destination_node_id"] == node_id and (link_name is None or link_name == relation["name"])
            )
        ]

        return nodes
    
    def get_nodes_linked_from(self, node, link_name = None):
        """get all nodes connected from node with node_id by link with link_name"""
        node_id = node["id"]

        nodes = [
            self.nodes[relation["destination_node_id"]] for relation in self.relations if (
                relation["source_node_id"] == node_id and (link_name is None or link_name == relation["name"])
            )
        ]

        return nodes
    
    def find_func_calls(self, func_calls: list[FuncCall]):
        func_call_full_names = set([func_call.get_full_name() for func_call in func_calls])
        return [model_node for model_node in self.model_nodes if model_node["name"] in func_call_full_names]
    
    @staticmethod
    def __validate_data(data):
        if not data:
            raise ValueError("corrupt ontology")
        
        if not ("nodes" in data):
            raise ValueError("corrupt ontology, nodes missing")
        
        if not ("relations" in data):
            raise ValueError("corrupt ontology, relations missing")
