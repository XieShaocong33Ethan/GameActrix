from __future__ import annotations


class CrowSphereError(Exception):
    pass


class ActionPlanError(CrowSphereError):
    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind


class RetryableActionPlanError(ActionPlanError):
    pass


class FatalActionPlanError(ActionPlanError):
    pass

