from typing import Annotated
from fastapi import APIRouter, Depends, Form, HTTPException, UploadFile, File
from huggingface_hub import HfApi, utils as huggingface_utils

from app.core.exceptions import create_exception_details
from app.models.onto.add_ml_task import AddMLTask
from app.models.onto.add_transformer_model import AddTransformerModel
from app.models.onto.generate_ontology import GenerateOntologyInput
from app.services.auth.auth import get_current_active_user
from app.services.onto.onto_generator import OntoGenerator

from ...services.onto import onto_parser as onto_parser_service
from ...services.onto.onto import MANAGEMENT_ONTO_PARSER

router = APIRouter(dependencies=[Depends(get_current_active_user)])


@router.post('/extract-knowledge-from-source-files')
async def extract_knowledge_from_source_files(source_files: Annotated[list[UploadFile], File(description="Python source code")]):
    knowledge_about_models = await onto_parser_service.extract_knowledge_from_source_files(MANAGEMENT_ONTO_PARSER, source_files)
    return knowledge_about_models


@router.get('/tree-view')
def get_tree_view():
    return MANAGEMENT_ONTO_PARSER.get_main_branches_tree_view()


@router.get('/models')
def get_models():
    return MANAGEMENT_ONTO_PARSER.get_model_nodes()


@router.get('/ml-tasks')
def get_ml_tasks():
    return MANAGEMENT_ONTO_PARSER.get_ml_tasks()


@router.post('/add-ml-task')
def add_ml_task(body: AddMLTask):
    MANAGEMENT_ONTO_PARSER.add_ml_task(
        new_node_name=body.new_node_name,
        parent_node_id=body.parent_node_id
    )
    return True


@router.post('/add-transformer-model')
def add_transformer_model(body: AddTransformerModel):

    api = HfApi()
    try:
        api.model_info(body.model_name_or_path)
    except huggingface_utils.HfHubHTTPError:
        raise HTTPException(status_code=400, detail=create_exception_details(f'Не удалось найти модель {body.model_name_or_path}'))
    
    MANAGEMENT_ONTO_PARSER.add_ml_transformer_model(
        parent_node_id=body.parent_node_id,
        node_name=body.node_name,
        model_name_or_path=body.model_name_or_path
    )
    return True


@router.post('/generate-ontology')
async def generate_ontology(
    python_version: str = Form(...),
    source_files: list[UploadFile] = File(description="Python source code"),
    version_file: UploadFile = File(description="requirements.txt file")
):
    onto_parser = await OntoGenerator.generate_ontology(source_files, version_file, python_version=python_version)
    data = onto_parser.get_update_raw_data()
    return data
