from fastapi import HTTPException, UploadFile
from app.constants.resources import BASIC_ONTO_PATH
from app.core.exceptions import create_exception_details
from app.services.onto.onto import MANAGEMENT_ONTO_PARSER
from app.services.onto.onto_parser import OntoParser, extract_knowledge_from_source_files
from app.services.onto.utils import read_uploaded_file


class OntoGenerator:
    @classmethod
    def get_basic_onto_parser(cls):
        return OntoParser.load_from_file(BASIC_ONTO_PATH, remove_history=True)
    
    @classmethod
    def get_node_by_name_or_create(cls, onto_parser: OntoParser, node_name: str, attrs: dict):
        node = onto_parser.get_node_by_name(node_name)
        if node is None:
            node = onto_parser.add_node(
                name=node_name,
                attributes=attrs
            )
        return node

    @classmethod
    async def generate_ontology(
        cls,
        source_files: list[UploadFile],
        version_file: UploadFile,
        python_version: str
    ):
        version_file_text = await read_uploaded_file(version_file)
        version_dict = dict()
        try:
            for line in version_file_text.split('\n'):
                line = line.strip()

                if not line:
                    continue

                lib_name, version = line.split('==')
                version_dict[lib_name] = version

        except Exception:
            raise HTTPException(status_code=400, detail=create_exception_details('Неверный файл версий'))

        onto_parser = cls.get_basic_onto_parser()

        machine_learning_node = onto_parser.get_node_by_name("Machine Learning")
        machine_learning_method_node = onto_parser.get_node_by_name("Machine Learning Method")
        python_lib_node = onto_parser.get_node_by_name("Python Library")

        python_node = onto_parser.get_node_by_name("Python")
        onto_parser.add_relation(
            name="version",
            source_node_id=python_node["id"],
            destination_node_id=onto_parser.add_node(name=python_version)["id"]
        )

        knowledge_about_models = await extract_knowledge_from_source_files(MANAGEMENT_ONTO_PARSER, source_files)

        versioned_lib_names = set()
        for knowledge_about_model in knowledge_about_models:
            first_lib_nodes = []
            for library in knowledge_about_model["libraries"]:
                first_lib_node, last_lib_node = cls.create_ml_lib_nodes(onto_parser, python_lib_node, library)
                if last_lib_node["name"] in version_dict and last_lib_node["name"] not in versioned_lib_names:
                    onto_parser.add_relation(
                        name="version",
                        source_node_id=last_lib_node["id"],
                        destination_node_id=onto_parser.add_node(name=version_dict[last_lib_node["name"]])["id"]
                    )
                    versioned_lib_names.add(last_lib_node["name"])
                first_lib_nodes.append(first_lib_node)
            ml_model_node = cls.create_ml_model_node(onto_parser, machine_learning_method_node, ml_model=knowledge_about_model, lib_nodes=first_lib_nodes)
            cls.create_ml_tasks(onto_parser, machine_learning_node, tasks=knowledge_about_model["tasks"], ml_model_node=ml_model_node)
        
        return onto_parser
    
    @classmethod
    def create_ml_lib_nodes(cls, onto_parser: OntoParser, python_lib_node: dict, library: dict[str]):
        first_node = None
        lib_name = library["name"]
        names = lib_name.split('.')

        last_node = None
        for i in range(len(names)):
            node_name = '.'.join(names[:len(names) - i])

            node = cls.get_node_by_name_or_create(
                onto_parser=onto_parser,
                node_name=node_name,
                attrs=library["attributes"]
            )

            if i == 0:
                first_node = node

            if last_node is not None:
                onto_parser.add_relation(name="a_part_of", source_node_id=last_node["id"], destination_node_id=node["id"])

            last_node = node
        
        onto_parser.add_relation(name="is_a", source_node_id=last_node["id"], destination_node_id=python_lib_node["id"])

        return first_node, last_node
    

    @classmethod
    def create_ml_model_node(cls, onto_parser: OntoParser, machine_learning_method_node: dict, ml_model: dict, lib_nodes: list[dict]):       
        ml_model_node = cls.get_node_by_name_or_create(
            onto_parser=onto_parser,
            node_name=ml_model["name"],
            attrs=ml_model["attributes"]
        )
        
        onto_parser.add_relation(
            name="is_a",
            source_node_id=ml_model_node["id"],
            destination_node_id=machine_learning_method_node["id"]
        )

        for lib_node in lib_nodes:
            onto_parser.add_relation(
                name="use",
                source_node_id=lib_node["id"],
                destination_node_id=ml_model_node["id"]
            )
        

        combination_1 = ml_model.get("combination_1")
        combination_2 = ml_model.get("combination_2")

        if combination_1 is not None and combination_2 is not None:
            combination_1_node = cls.get_node_by_name_or_create(
                onto_parser=onto_parser,
                node_name=combination_1["name"],
                attrs=combination_1["attributes"]
            )
            combination_2_node = cls.get_node_by_name_or_create(
                onto_parser=onto_parser,
                node_name=combination_2["name"],
                attrs=combination_2["attributes"]
            )

            onto_parser.add_relation(
                name="is_a",
                source_node_id=combination_1_node["id"],
                destination_node_id=machine_learning_method_node["id"]
            )
            
            onto_parser.add_relation(
                name="is_a",
                source_node_id=combination_2_node["id"],
                destination_node_id=machine_learning_method_node["id"]
            )

            onto_parser.add_relation(
                name="combination",
                source_node_id=combination_1_node["id"],
                destination_node_id=ml_model_node["id"]
            )
            onto_parser.add_relation(
                name="with",
                source_node_id=ml_model_node["id"],
                destination_node_id=combination_2_node["id"]
            )

            if len(lib_nodes) >= 1:
                onto_parser.add_relation(
                    name="use",
                    source_node_id=lib_nodes[0]["id"],
                    destination_node_id=combination_1_node["id"]
                )
            
            if len(lib_nodes) >= 2:
                onto_parser.add_relation(
                    name="use",
                    source_node_id=lib_nodes[1]["id"],
                    destination_node_id=combination_2_node["id"]
                )
        
        return ml_model_node

    @classmethod
    def create_ml_tasks(cls, onto_parser: OntoParser, machine_learning_node: dict, tasks: list[dict[str, str]], ml_model_node: dict):
        if len(tasks) < 2:
            return

        first_task_node = None
        last_task_node = None
        for index, task in enumerate(tasks[1:][::-1]):
            task_node = cls.get_node_by_name_or_create(onto_parser, node_name=task["name"], attrs=task["attributes"])
            if index == 0:
                first_task_node = task_node
                onto_parser.add_relation(
                    name="used_for",
                    source_node_id=ml_model_node["id"],
                    destination_node_id=task_node["id"]
                )
            
            if last_task_node is not None:
                onto_parser.add_relation(
                    name="is_a",
                    source_node_id=last_task_node["id"],
                    destination_node_id=task_node["id"]
                )

            last_task_node = task_node
        
        onto_parser.add_relation(
            name="is_a",
            source_node_id=last_task_node["id"],
            destination_node_id=machine_learning_node["id"]
        )
    
        return first_task_node, last_task_node
