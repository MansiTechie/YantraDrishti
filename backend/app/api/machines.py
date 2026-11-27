from fastapi import APIRouter, Depends, HTTPException, status

from app.repositories.machine_repository import (
    DatabaseUnavailableError,
    DuplicateMachineError,
    MachineRepository,
    get_machine_repository,
)
from app.schemas.machine import MachineState

router = APIRouter(
    prefix="/api/v1/machines",
    tags=["Machines"],
)


@router.post(
    "",
    response_model=MachineState,
    status_code=status.HTTP_201_CREATED,
    responses={
        409: {"description": "Machine already exists"},
        503: {"description": "Database service unavailable"},
    },
)
def register_machine(
    machine: MachineState,
    repo: MachineRepository = Depends(get_machine_repository),
):
    try:
        created_model = repo.create(machine)
        return created_model.to_schema()
    except DuplicateMachineError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e),
        )
    except DatabaseUnavailableError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e),
        )


@router.get(
    "",
    response_model=list[MachineState],
    responses={
        503: {"description": "Database service unavailable"},
    },
)
def get_all_machines(
    repo: MachineRepository = Depends(get_machine_repository),
):
    try:
        models = repo.get_all()
        return [m.to_schema() for m in models]
    except DatabaseUnavailableError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e),
        )


@router.get(
    "/{machine_id}",
    response_model=MachineState,
    responses={
        404: {"description": "Machine not found"},
        503: {"description": "Database service unavailable"},
    },
)
def get_machine(
    machine_id: str,
    repo: MachineRepository = Depends(get_machine_repository),
):
    try:
        model = repo.get_by_id(machine_id)
        if model is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Machine '{machine_id}' not found",
            )
        return model.to_schema()
    except DatabaseUnavailableError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e),
        )
