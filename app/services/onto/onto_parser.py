import json
import networkx as nx

from ...models.parser.python_parser import FuncCall


class OntoParser:
    main_branches = ["Task", "Method", "Experiment", "Software"]

    def __init__(self, data):
        self.last_id = data["last_id"]
        self.namespaces = data["namespaces"]
        self.nodes = {node["id"]: node for node in data["nodes"]}
        self.relations = data["relations"]
        self.digraph = self.create_digraph()
        self.leaves = [node for node in data["nodes"] if node["attributes"].get("leaf", False)]

        self.model_nodes = []
        for model_node in self.get_nodes_linked_to(self.get_node_by_name("Machine Learning Method"), ["is_a"]):
            self.model_nodes.extend(self.get_nodes_linked_to(model_node, ["use"]))

    def create_digraph(self):
        graph = nx.DiGraph()
        for node_id in self.nodes:
            graph.add_node(node_id)
        for link in self.relations:
            graph.add_edge(link["source_node_id"], link["destination_node_id"])
        return graph
    
    def shortest_path(self, _from, _to):
        shortest_path = nx.shortest_path(nx.to_undirected(self.digraph), _from["id"], _to["id"])
        shortest_path = [self.get_node_by_id(node_id) for node_id in shortest_path]
        return shortest_path

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
    
    def get_main_branch_nodes(self):
        return [self.get_node_by_name(name) for name in self.main_branches]

    def get_leaf_nodes_for_branch(self, branch_name: str):
        return [leaf for leaf in self.leaves if leaf["attributes"]["branch"] == branch_name]
    
    def get_main_branches_tree_view(self):
        main_branche_nodes = self.get_main_branch_nodes()

        node_paths = []
        for branch_node in main_branche_nodes:
            leaf_nodes = self.get_leaf_nodes_for_branch(branch_node["name"])
            shortest_paths = [self.shortest_path(branch_node, leaf_node) for leaf_node in leaf_nodes]
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
        func_call_full_names = set([func_call.get_full_name() for func_call in func_calls])
        return [model_node for model_node in self.model_nodes if model_node["name"] in func_call_full_names]
    
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

def main():
    onto_parser = OntoParser.load_from_file(r"G:\PythonProjects\ExperimentApp\ExperimentApi\resources\wine-recognition.ont")
    main_branches = onto_parser.get_main_branches_tree_view()


if __name__ == '__main__':
    main()
