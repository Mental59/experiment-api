from typing import Annotated
from fastapi import APIRouter, Depends, UploadFile, File

from app.models.onto.add_ml_task import AddMLTask
from app.models.onto.add_transformer_model import AddTransformerModel
from app.services.auth.auth import get_current_active_user

from ...services.onto import onto_parser as onto_parser_service
from ...services.onto.onto import ONTO_PARSER

router = APIRouter(dependencies=[Depends(get_current_active_user)])


@router.post('/find-models')
async def find_models(source_files: Annotated[list[UploadFile], File(description="Python source code")]):
    models = await onto_parser_service.find_models(ONTO_PARSER, source_files)
    return models


@router.get('/tree-view')
def get_tree_view():
    return ONTO_PARSER.get_main_branches_tree_view()


@router.get('/models')
def get_models():
    return ONTO_PARSER.get_model_nodes()


@router.get('/ml-tasks')
def get_ml_tasks():
    return ONTO_PARSER.get_ml_tasks()


@router.post('/add-ml-task')
def add_ml_task(body: AddMLTask):
    ONTO_PARSER.add_ml_task(
        new_node_name=body.new_node_name,
        parent_node_id=body.parent_node_id
    )
    return True


@router.post('/add-transformer-model')
def add_transformer_model(body: AddTransformerModel):
    ONTO_PARSER.add_ml_transformer_model(
        parent_node_id=body.parent_node_id,
        node_name=body.node_name,
        model_name_or_path=body.model_name_or_path
    )
    return True
