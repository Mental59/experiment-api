import ast


class Import:
    def __init__(self, name, asname, module = None):
        self.imported_from = module
        self.imported_name = name
        self.imported_asname = asname
    
    def get_key(self):
        return self.imported_asname if self.imported_asname is not None else self.imported_name
    
    def get_full_name(self):
        if self.imported_from is None:
            return self.imported_name
        return f"{self.imported_from}.{self.imported_name}"
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(imported_from={self.imported_from}, imported_name={self.imported_name}, imported_asname={self.imported_asname})"


class FuncArg:
    def __init__(self, arg: ast.expr, index: int, keyword: str | None = None, value = None):
        self.arg = arg
        self.index = index
        self.keyword = keyword
        self.value = value  
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(arg={self.arg}, value={self.value}, index={self.index}, keyword={self.keyword})"


class FuncCall:
    def __init__(self, name: str, args: list[FuncArg], module_name: str | None=None):
        self.func_name = name
        self.module_name = module_name
        self.args = args
        self.import_module = None
    
    def set_import_module(self, import_module):
        self.import_module = import_module
    
    def get_full_name(self):
        if self.import_module is None:
            return self.func_name
        if self.import_module.imported_name == self.func_name:
            return self.import_module.get_full_name()
        
        return f"{self.import_module.get_full_name()}.{self.func_name}"
    
    def get_arg_by_keyword_or_index(self, index: int, keyword: str | None = None) -> FuncArg | None:
        if keyword is not None:
            for arg in self.args:
                if arg.keyword == keyword:
                    return arg
        
        if index < len(self.args):
            return self.args[index]
 
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(func_name={self.func_name}, module_name={self.module_name}, import_module={self.import_module}, args={self.args})"
