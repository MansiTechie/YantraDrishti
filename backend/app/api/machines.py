
from fastapi import APIRouter, HTTPException, status

from app.schemas.machine import MachineState

router = APIRouter(
    prefix="/api/v1/machines",
    tags=["Machines"],
)

# Temporary in-memory storage
machines: dict[str, MachineState] = {}


@router.post(
    "",
    response_model=MachineState,
    status_code=status.HTTP_201_CREATED,
    responses={
        409: {"description": "Machine already exists"},
    },
)
def register_machine(machine: MachineState):
    machine_id = machine.identity.machine_id

    if machine_id in machines:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Machine '{machine_id}' already exists",
        )

    machines[machine_id] = machine
    return machine


@router.get(
    "",
    response_model=list[MachineState],
)
def get_all_machines():
    return list(machines.values())


@router.get(
    "/{machine_id}",
    response_model=MachineState,
    responses={
        404: {"description": "Machine not found"},
    },
)
def get_machine(machine_id: str):
    machine = machines.get(machine_id)

    if machine is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Machine '{machine_id}' not found",
        )

    return machine
