from fastapi import APIRouter

from . import datasets, root

router = APIRouter()
router.include_router(datasets.router, tags=['dataset'], prefix='/datasets')
router.include_router(root.router, tags=['root'])
