"""
Quick Reference: grpc_server_utils.py

Copy-paste examples for common patterns in gRPC game server implementation.
"""

# ============================================================================
# IMPORTS
# ============================================================================

from evaluation_utils.grpc_server_utils import (
    SessionManager,
    IdempotencyTracker,
    require_session,
    require_lock,
    validate_session_and_acquire_lock,
    SESSION_TIMEOUT_SECONDS
)
import threading
import grpc


# ============================================================================
# PATTERN 1: Basic Session Management
# ============================================================================

class BasicGameServer:
    """Minimal session management example."""

    def __init__(self):
        self.session_mgr = SessionManager()

    def RegisterSession(self, request, context):
        # Check if session already active
        if self.session_mgr._token and not self.session_mgr.is_expired():
            context.abort(
                grpc.StatusCode.PERMISSION_DENIED,
                "Server occupied by another client"
            )

        # Create new session
        token = self.session_mgr.register()
        return SessionResponse(session_token=token)

    @require_session(self.session_mgr)
    def GetGameConfig(self, request, context):
        # request.session_token already validated
        # session automatically touched to extend lifetime
        return GameConfig(game_id="my_game")


# ============================================================================
# PATTERN 2: Fail-Fast Lock for Single Operation
# ============================================================================

class SingleLockGameServer:
    """Single lock protecting critical operations."""

    def __init__(self):
        self.session_mgr = SessionManager()
        self.action_lock = threading.Lock()

    @require_session(self.session_mgr)
    @require_lock(self.action_lock, "GetObservation")
    def GetObservation(self, request, context):
        # Session validated + Lock acquired
        # Lock automatically released in finally block
        obs_str, obs_image, info = self._game.load_current_obs()
        return Observation(obs_text=obs_str, obs_image=obs_image, info=info)


# ============================================================================
# PATTERN 3: Step with Idempotency
# ============================================================================

class IdempotentStepServer:
    """Step operation with idempotency tracking."""

    def __init__(self):
        self.session_mgr = SessionManager()
        self.action_lock = threading.Lock()
        self.idempotency = IdempotencyTracker()
        self.last_result = None  # Cache for duplicate requests

    @require_session(self.session_mgr)
    @require_lock(self.action_lock, "Step")
    def Step(self, request, context):
        # Check for duplicate request
        if request.request_id and self.idempotency.is_duplicate(request.request_id):
            # Return cached result
            if self.last_result:
                return self.last_result

        # Record request_id BEFORE execution
        if request.request_id:
            self.idempotency.record(request.request_id)

        # Execute action
        try:
            score, is_finished, _ = self._game.dispatch_action(request.action)
            obs_str, obs_image, info = self._game.load_current_obs()

            result = StepResult(
                score=score,
                is_finished=is_finished,
                avg_score=self._game.avg_score,
                observation=Observation(
                    obs_text=obs_str,
                    obs_image=obs_image,
                    info=info
                )
            )

            # Cache result for idempotency
            self.last_result = result
            return result

        except Exception as e:
            context.set_details(f"Step failed: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            raise


# ============================================================================
# PATTERN 4: Combined Decorator (Convenience)
# ============================================================================

class ConvenienceDecoratorServer:
    """Using combined session + lock decorator."""

    def __init__(self):
        self.session_mgr = SessionManager()
        self.action_lock = threading.Lock()

    @validate_session_and_acquire_lock(
        self.session_mgr,
        self.action_lock,
        "Step"
    )
    def Step(self, request, context):
        # Both session validated AND lock acquired in one decorator
        return self._execute_step(request.action)


# ============================================================================
# PATTERN 5: Manual Session Validation (Advanced)
# ============================================================================

class ManualValidationServer:
    """Manual session validation without decorator."""

    def __init__(self):
        self.session_mgr = SessionManager()

    def CustomMethod(self, request, context):
        # Manual validation for special cases
        token = request.session_token

        if not self.session_mgr.validate(token):
            if self.session_mgr.is_expired():
                context.abort(
                    grpc.StatusCode.UNAUTHENTICATED,
                    "Session expired"
                )
            else:
                context.abort(
                    grpc.StatusCode.UNAUTHENTICATED,
                    "Invalid session token"
                )

        # Touch session to extend lifetime
        self.session_mgr.touch()

        # Proceed with operation
        return perform_custom_operation()


# ============================================================================
# PATTERN 6: Manual Lock Acquisition (Advanced)
# ============================================================================

class ManualLockServer:
    """Manual lock acquisition for complex logic."""

    def __init__(self):
        self.action_lock = threading.Lock()

    def ComplexMethod(self, request, context):
        # Try to acquire lock fail-fast
        acquired = self.action_lock.acquire(blocking=False)

        if not acquired:
            current_thread = threading.current_thread()
            context.abort(
                grpc.StatusCode.ABORTED,
                f"Busy: Operation in progress (thread: {current_thread.name})"
            )

        try:
            # Perform complex operation with multiple steps
            self._step1()
            if some_condition:
                self._step2()
            else:
                self._step3()
            return result

        except Exception as e:
            # Log error with context
            logger.error(f"Operation failed: {e}", exc_info=True)
            context.set_details(f"Operation failed: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            raise

        finally:
            # ALWAYS release lock
            self.action_lock.release()


# ============================================================================
# PATTERN 7: Complete Production Server
# ============================================================================

class ProductionGameServer:
    """
    Complete example with all best practices.

    Features:
    - Session management with timeout
    - Fail-fast lock acquisition
    - Idempotency tracking
    - Comprehensive error handling
    - Thread-safe state management
    """

    def __init__(self, game_instance):
        # Core game instance
        self._game = game_instance

        # Session management
        self.session_mgr = SessionManager(timeout_seconds=SESSION_TIMEOUT_SECONDS)

        # Concurrency control
        self.action_lock = threading.Lock()

        # Idempotency tracking
        self.idempotency = IdempotencyTracker()
        self.last_step_result = None

    # ------------------------------------------------------------------------
    # RegisterSession: No decorators (creates session)
    # ------------------------------------------------------------------------

    def RegisterSession(self, request, context):
        """Create new session token."""
        # Reject if session exists and not expired
        if self.session_mgr._token and not self.session_mgr.is_expired():
            context.abort(
                grpc.StatusCode.PERMISSION_DENIED,
                "Server occupied by another client. Wait for timeout or use existing session."
            )

        # Register new session
        token = self.session_mgr.register()
        logger.info(f"New session registered: {token}")

        # Reset idempotency state
        self.last_step_result = None

        return SessionResponse(session_token=token)

    # ------------------------------------------------------------------------
    # GetGameConfig: Session validation only
    # ------------------------------------------------------------------------

    @require_session(self.session_mgr)
    def GetGameConfig(self, request, context):
        """Get game configuration and current progress."""
        config = self._game.get_game_config()

        return GameConfig(
            game_id=config["game_id"],
            max_steps=config["max_steps"],
            max_episodes=config["max_episodes"],
            current_episode=config.get("current_episode", 0),
            current_step=config.get("current_step", 0)
        )

    # ------------------------------------------------------------------------
    # GetObservation: Session + Lock (read operation)
    # ------------------------------------------------------------------------

    @require_session(self.session_mgr)
    @require_lock(self.action_lock, "GetObservation")
    def GetObservation(self, request, context):
        """Get current observation without advancing game state."""
        try:
            obs_str, obs_image_bytes, game_info = self._game.load_current_obs()

            return Observation(
                obs_text=obs_str,
                obs_image=obs_image_bytes,
                info={k: str(v) for k, v in game_info.items()}
            )

        except Exception as e:
            logger.error(f"GetObservation failed: {e}", exc_info=True)
            context.set_details(f"Failed to get observation: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            raise

    # ------------------------------------------------------------------------
    # Step: Session + Lock + Idempotency (write operation)
    # ------------------------------------------------------------------------

    @require_session(self.session_mgr)
    @require_lock(self.action_lock, "Step")
    def Step(self, request, context):
        """Execute action and return result with new observation."""
        # Idempotency check
        if request.request_id and self.idempotency.is_duplicate(request.request_id):
            if self.last_step_result:
                logger.info(f"Returning cached result for duplicate request: {request.request_id}")
                return self.last_step_result

        # Record request_id BEFORE execution
        if request.request_id:
            self.idempotency.record(request.request_id)

        try:
            # Execute action
            score, is_finished, avg_score = self._game.dispatch_action_and_get_score(
                request.action
            )

            # Get new observation after step
            obs_str, obs_image_bytes, game_info = self._game.load_current_obs()

            # Build result
            result = StepResult(
                score=score,
                is_finished=is_finished,
                avg_score=avg_score,
                observation=Observation(
                    obs_text=obs_str,
                    obs_image=obs_image_bytes,
                    info={k: str(v) for k, v in game_info.items()}
                )
            )

            # Cache for idempotency
            self.last_step_result = result

            logger.debug(
                f"Step executed: action={request.action}, "
                f"score={score}, finished={is_finished}"
            )

            return result

        except Exception as e:
            logger.error(f"Step failed: {e}", exc_info=True)
            context.set_details(f"Step execution failed: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            raise


# ============================================================================
# ERROR HANDLING CHEATSHEET
# ============================================================================

"""
Common gRPC Status Codes:

grpc.StatusCode.UNAUTHENTICATED
- Session token invalid/expired/missing
- Client should call RegisterSession

grpc.StatusCode.PERMISSION_DENIED
- Server occupied by another client
- Client should wait for timeout or disconnect other client

grpc.StatusCode.ABORTED
- Lock contention (operation already in progress)
- Client should retry after brief delay (100ms)

grpc.StatusCode.INTERNAL
- Server-side error during execution
- Check server logs for details

grpc.StatusCode.UNAVAILABLE
- Server not reachable
- Client should retry with backoff
"""


# ============================================================================
# CLIENT RETRY PATTERN
# ============================================================================

def client_call_with_retry(stub_method, request, max_retries=3):
    """
    Example client-side retry pattern.

    Handles:
    - ABORTED: Retry once after 100ms (lock contention)
    - UNAUTHENTICATED: Re-register session and retry
    - Other errors: Propagate
    """
    import time

    for attempt in range(max_retries):
        try:
            return stub_method(request)

        except grpc.RpcError as e:
            status_code = e.code()

            if status_code == grpc.StatusCode.ABORTED:
                # Lock contention - retry after brief delay
                if attempt < max_retries - 1:
                    time.sleep(0.1)
                    continue
                else:
                    raise

            elif status_code == grpc.StatusCode.UNAUTHENTICATED:
                # Session expired - re-register
                response = stub.RegisterSession(Empty())
                session_token = response.session_token
                # Update request with new token and retry
                request.session_token = session_token
                continue

            else:
                # Other errors - don't retry
                raise

    raise RuntimeError("Max retries exceeded")
