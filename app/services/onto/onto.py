import os

from .onto_parser import OntoParser
from ...constants.resources import RESOURCES_PATH

ONTO_PATH = os.path.join(RESOURCES_PATH, 'wine-recognition.ont')
ONTO_PARSER = OntoParser.load_from_file(ONTO_PATH)
