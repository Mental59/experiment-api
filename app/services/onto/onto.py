from app.constants.resources import ONTO_PATH
from .onto_parser import OntoParser

ONTO_PARSER = OntoParser.load_from_file(ONTO_PATH)
