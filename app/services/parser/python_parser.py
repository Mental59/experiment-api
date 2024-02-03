import ast

from ...models.parser.python_parser import FuncCall, Import


def parse(source: str):
    """ parses python source code and returns func calls with modules from which they were imported """
    func_calls: list[FuncCall] = []
    imported_names: list[Import] = []

    tree = ast.parse(source)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_names.extend([Import(name.name, name.asname) for name in node.names])
        if isinstance(node, ast.ImportFrom):
            if node.level == 0:
                imported_names.extend([Import(name.name, name.asname, node.module) for name in node.names])

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                func_calls.append(FuncCall(node.func.attr, node.func.value.id))
            if isinstance(node.func, ast.Name):
                func_calls.append(FuncCall(node.func.id))
    
    imported_names_dict = {name.get_key(): name for name in imported_names}

    for func_call in func_calls:
        imported_name = imported_names_dict.get(func_call.module_name if func_call.module_name is not None else func_call.func_name)
        if imported_name is not None:
            func_call.set_import_module(imported_name)

    return func_calls
