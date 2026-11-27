import logging
from typing import Optional
from fastapi import Depends
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.machine import MachineModel
from app.schemas.machine import MachineState

logger = logging.getLogger(__name__)


class MachineRepositoryError(Exception):
    """Base exception for machine repository operations."""
    pass


class DuplicateMachineError(MachineRepositoryError):
    """Raised when attempting to create a machine that already exists."""
    pass


class MachineNotFoundError(MachineRepositoryError):
    """Raised when a requested machine cannot be found."""
    pass


class DatabaseUnavailableError(MachineRepositoryError):
    """Raised when database connection fails or is unreachable."""
    pass


class MachineRepository:
    """
    Repository layer for MachineModel database operations.
    Encapsulates persistence logic and separates SQLAlchemy queries from API routes.
    """

    def __init__(self, db: Session):
        self.db = db

    def get_by_id(self, machine_id: str) -> Optional[MachineModel]:
        """
        Retrieves a single machine by unique machine_id.
        Returns None if machine does not exist.
        """
        try:
            stmt = select(MachineModel).where(MachineModel.machine_id == machine_id)
            return self.db.scalars(stmt).first()
        except OperationalError as exc:
            logger.error("Database connection failure in get_by_id: %s", exc)
            raise DatabaseUnavailableError(
                "Database connection failed. Verify database service is running and DATABASE_URL is configured."
            ) from exc

    def get_all(self, skip: int = 0, limit: int = 100) -> list[MachineModel]:
        """
        Retrieves paginated list of machines.
        """
        try:
            stmt = select(MachineModel).offset(skip).limit(limit)
            return list(self.db.scalars(stmt).all())
        except OperationalError as exc:
            logger.error("Database connection failure in get_all: %s", exc)
            raise DatabaseUnavailableError(
                "Database connection failed. Verify database service is running and DATABASE_URL is configured."
            ) from exc

    def create(self, machine_state: MachineState) -> MachineModel:
        """
        Creates and persists a new MachineModel from a validated MachineState.
        Raises DuplicateMachineError if machine_id already exists.
        """
        machine_id = machine_state.identity.machine_id

        # Pre-check for existing record
        existing = self.get_by_id(machine_id)
        if existing is not None:
            raise DuplicateMachineError(f"Machine '{machine_id}' already exists")

        model = MachineModel.from_schema(machine_state)
        try:
            self.db.add(model)
            self.db.commit()
            self.db.refresh(model)
            return model
        except IntegrityError as exc:
            self.db.rollback()
            raise DuplicateMachineError(f"Machine '{machine_id}' already exists") from exc
        except OperationalError as exc:
            self.db.rollback()
            logger.error("Database connection failure in create: %s", exc)
            raise DatabaseUnavailableError(
                "Database connection failed. Verify database service is running and DATABASE_URL is configured."
            ) from exc
        except Exception:
            self.db.rollback()
            raise

    def update(self, machine_id: str, machine_state: MachineState) -> MachineModel:
        """
        Updates an existing machine record from a validated MachineState.
        Raises MachineNotFoundError if machine does not exist.
        """
        model = self.get_by_id(machine_id)
        if model is None:
            raise MachineNotFoundError(f"Machine '{machine_id}' not found")

        try:
            model.update_from_schema(machine_state)
            self.db.commit()
            self.db.refresh(model)
            return model
        except OperationalError as exc:
            self.db.rollback()
            logger.error("Database connection failure in update: %s", exc)
            raise DatabaseUnavailableError(
                "Database connection failed. Verify database service is running and DATABASE_URL is configured."
            ) from exc
        except Exception:
            self.db.rollback()
            raise

    def delete(self, machine_id: str) -> bool:
        """
        Deletes a machine record by machine_id.
        Returns True if deleted, False if machine was not found.
        """
        model = self.get_by_id(machine_id)
        if model is None:
            return False

        try:
            self.db.delete(model)
            self.db.commit()
            return True
        except OperationalError as exc:
            self.db.rollback()
            logger.error("Database connection failure in delete: %s", exc)
            raise DatabaseUnavailableError(
                "Database connection failed. Verify database service is running and DATABASE_URL is configured."
            ) from exc
        except Exception:
            self.db.rollback()
            raise


def get_machine_repository(db: Session = Depends(get_db)) -> MachineRepository:
    """
    FastAPI dependency that provides an instantiated MachineRepository.
    """
    return MachineRepository(db)
