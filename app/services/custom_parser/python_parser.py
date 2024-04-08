import ast

if __name__ == '__main__':
    import sys
    import os
    sys.path.insert(0, os.getcwd())

from app.models.custom_parser.python_parser import FuncCall, Import, FuncArg


def parse_func_args(
    args: list[ast.expr],
    keywords: list[ast.keyword],
    var_assignments_dict: dict[str]
) -> list[FuncArg]:
    def extract_value_from_expr(expr: ast.expr):
        if isinstance(expr, ast.Constant):
            return expr.value
        elif isinstance(expr, ast.Name):
            return var_assignments_dict.get(expr.id, None)
        return None
    
    func_args = [FuncArg(arg=arg, index=index, value=extract_value_from_expr(arg)) for index, arg in enumerate(args)]
    func_keywords = [FuncArg(arg=keyword.value, index=index + len(func_args), keyword=keyword.arg, value=extract_value_from_expr(keyword.value)) for index, keyword in enumerate(keywords)]
    return func_args + func_keywords


def parse(source: str):
    """ parses python source code and returns func calls with modules from which they were imported """
    func_calls: list[FuncCall] = []
    imported_names: list[Import] = []

    var_assignments_dict = dict()

    tree = ast.parse(source)

    for assignment in filter(lambda node: isinstance(node, ast.Assign), ast.walk(tree)):
        variables: list[ast.Name] = list(filter(lambda var: isinstance(var, ast.Name), assignment.targets))
        var_value = None

        if len(variables) == 0:
            continue

        if isinstance(assignment.value, ast.Constant):
            var_value = assignment.value.value
        elif isinstance(assignment.value, ast.Name):
            var_value = var_assignments_dict.get(assignment.value.id, None)
        
        if var_value is None:
            continue

        for variable in variables:
            var_assignments_dict[variable.id] = var_value

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_names.extend([Import(name.name, name.asname) for name in node.names])
        if isinstance(node, ast.ImportFrom):
            if node.level == 0:
                imported_names.extend([Import(name.name, name.asname, node.module) for name in node.names])
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                func_calls.append(
                    FuncCall(
                        name=node.func.attr,
                        module_name=node.func.value.id,
                        args=parse_func_args(args=node.args, keywords=node.keywords, var_assignments_dict=var_assignments_dict)
                    )
                )
            if isinstance(node.func, ast.Name):
                func_calls.append(
                    FuncCall(
                        name=node.func.id,
                        args=parse_func_args(args=node.args, keywords=node.keywords, var_assignments_dict=var_assignments_dict)
                    )
                )

    imported_names_dict = {name.get_key(): name for name in imported_names}

    for func_call in func_calls:
        imported_name = imported_names_dict.get(func_call.module_name if func_call.module_name is not None else func_call.func_name)
        if imported_name is not None:
            func_call.set_import_module(imported_name)

    return func_calls


if __name__ == '__main__':
    with open(r"G:\PythonProjects\WineRecognition2\nn\model\bert_different_calls.py") as file:
        func_calls = parse(file.read())
    print('\n'.join(map(lambda func_call: f'{func_call.get_full_name()}: args={func_call.args}', func_calls)))
