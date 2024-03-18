from fastapi import APIRouter

from . import datasets, onto_parser, root, ml, experiments, auth

router = APIRouter()
router.include_router(datasets.router, tags=['dataset'], prefix='/datasets')
router.include_router(root.router, tags=['root'])
router.include_router(ml.router, tags=['machine learning'], prefix='/ml')
router.include_router(experiments.router, tags=['experiments'], prefix='/exp')
router.include_router(onto_parser.router, tags=['parser'], prefix='/onto-parser')
router.include_router(auth.router, tags=['auth'], prefix='/auth')
