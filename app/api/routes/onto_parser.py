from typing import Annotated
import os

from fastapi import APIRouter, UploadFile, File

from ...services.onto import onto_parser as onto_parser_service
from ...services.onto.onto import ONTO_PARSER

router = APIRouter()

@router.post('/find-models')
async def find_models(source_files: Annotated[list[UploadFile], File(description="Python source code")]):
    models = await onto_parser_service.find_models(ONTO_PARSER, source_files)
    return models


@router.get('/tree-view')
def get_tree_view():
    return ONTO_PARSER.get_main_branches_tree_view()
