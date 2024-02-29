from typing import Annotated
import os

from fastapi import APIRouter, UploadFile, File

from ...services.onto import onto_parser as onto_parser_service
from ...constants.resources import RESOURCES_PATH

router = APIRouter()

onto = onto_parser_service.OntoParser.load_from_file(os.path.join(RESOURCES_PATH, 'wine-recognition.ont'))

@router.post('/find-models')
async def find_models(source_files: Annotated[list[UploadFile], File(description="Python source code")]):
    models = await onto_parser_service.find_models(onto, source_files)
    return models


@router.get('/tree-view')
def get_tree_view():
    return onto.get_main_branches_tree_view()
