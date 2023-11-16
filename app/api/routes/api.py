from fastapi import APIRouter

from . import datasets, root, ml

router = APIRouter()
router.include_router(datasets.router, tags=['dataset'], prefix='/datasets')
router.include_router(root.router, tags=['root'])
router.include_router(ml.router, tags=['machine learning'], prefix='/ml')
