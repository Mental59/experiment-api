from typing import Annotated
import os

from fastapi import APIRouter, UploadFile, File, HTTPException

from ...services.onto.onto_parser import OntoParser
from ...services.parser.python_parser import parse as parse_python_code
from ...constants.resources import RESOURCES_PATH
from ...core.exceptions import create_exception_details

router = APIRouter()

onto = OntoParser.load_from_file(os.path.join(RESOURCES_PATH, 'wine-recognition.ont'))

@router.post('/find-models')
async def find_models(source_file: Annotated[UploadFile, File(description="Python source code")]):
    try:
        content_bytes = await source_file.read()
        text = content_bytes.decode("utf-8")
    except ValueError as error:
        raise HTTPException(400, detail=create_exception_details(f"Invalid source file; reason: {error}"))
    
    return onto.find_func_calls(parse_python_code(text))
    