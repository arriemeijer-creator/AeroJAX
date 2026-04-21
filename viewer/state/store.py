"""
Redux-style state management for the CFD viewer.
Provides unidirectional data flow with actions, reducers, and a central store.
"""

from typing import Callable, Dict, Any, TypedDict, Optional
from dataclasses import dataclass, field, replace
from enum import Enum


# ============================================================================
# Action Types
# ============================================================================

class ActionType(Enum):
    """Enumeration of all possible action types"""
    SET_OBSTACLE_TYPE = "SET_OBSTACLE_TYPE"
    SET_NACA_AIRFOIL = "SET_NACA_AIRFOIL"
    SET_NACA_CHORD = "SET_NACA_CHORD"
    SET_NACA_ANGLE = "SET_NACA_ANGLE"
    SET_NACA_X = "SET_NACA_X"
    SET_NACA_Y = "SET_NACA_Y"
    SET_CYLINDER_RADIUS = "SET_CYLINDER_RADIUS"
    SET_CYLINDER_CENTER_X = "SET_CYLINDER_CENTER_X"
    SET_CYLINDER_CENTER_Y = "SET_CYLINDER_CENTER_Y"
    SET_OBSTACLE_POSITION = "SET_OBSTACLE_POSITION"
    # Simulation state actions
    SET_REYNOLDS_NUMBER = "SET_REYNOLDS_NUMBER"
    SET_U_INF = "SET_U_INF"
    SET_NU = "SET_NU"
    SET_GRID_NX = "SET_GRID_NX"
    SET_GRID_NY = "SET_GRID_NY"


@dataclass
class Action:
    """Represents a state change action"""
    type: ActionType
    payload: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# State Structure
# ============================================================================

@dataclass
class ObstacleState:
    """State related to obstacle configuration"""
    obstacle_type: str = 'naca_airfoil'  # 'cylinder', 'naca_airfoil', 'cow', 'three_cylinder_array'
    
    # NACA airfoil parameters
    naca_airfoil: str = 'NACA 0012'
    naca_chord: float = 0.45
    naca_angle: float = 10.0
    naca_x: float = 5.0
    naca_y: float = 2.5  # Center of domain (ly=5.0)
    
    # Cylinder parameters
    cylinder_radius: float = 0.18
    cylinder_center_x: float = 2.5
    cylinder_center_y: float = 1.875
    
    # Cow parameters
    cow_x: float = 5.0
    cow_y: float = 1.75  # 35% of default_ly=5.0
    
    # Three cylinder array parameters
    three_cylinder_x: float = 5.0
    three_cylinder_y: float = 10.0875


@dataclass
class SimulationState:
    """State related to simulation parameters"""
    reynolds_number: float = 2000.0
    u_inf: float = 1.0
    nu: float = 0.003
    grid_nx: int = 512
    grid_ny: int = 96


@dataclass
class AppState:
    """Main application state"""
    obstacle: ObstacleState = field(default_factory=ObstacleState)
    simulation: SimulationState = field(default_factory=SimulationState)
    # Future state slices can be added here:
    # ui: UIState
    # visualization: VisualizationState


# ============================================================================
# Reducers
# ============================================================================

def obstacle_reducer(state: ObstacleState, action: Action) -> ObstacleState:
    """
    Reducer for obstacle-related state changes.
    
    Args:
        state: Current obstacle state
        action: Action to process
        
    Returns:
        New obstacle state
    """
    if action.type == ActionType.SET_OBSTACLE_TYPE:
        return replace(state, obstacle_type=action.payload.get('obstacle_type', state.obstacle_type))
    
    elif action.type == ActionType.SET_NACA_AIRFOIL:
        return replace(state, naca_airfoil=action.payload.get('naca_airfoil', state.naca_airfoil))
    
    elif action.type == ActionType.SET_NACA_CHORD:
        return replace(state, naca_chord=action.payload.get('naca_chord', state.naca_chord))
    
    elif action.type == ActionType.SET_NACA_ANGLE:
        return replace(state, naca_angle=action.payload.get('naca_angle', state.naca_angle))
    
    elif action.type == ActionType.SET_NACA_X:
        return replace(state, naca_x=action.payload.get('naca_x', state.naca_x))
    
    elif action.type == ActionType.SET_NACA_Y:
        return replace(state, naca_y=action.payload.get('naca_y', state.naca_y))
    
    elif action.type == ActionType.SET_CYLINDER_RADIUS:
        return replace(state, cylinder_radius=action.payload.get('cylinder_radius', state.cylinder_radius))
    
    elif action.type == ActionType.SET_CYLINDER_CENTER_X:
        return replace(state, cylinder_center_x=action.payload.get('cylinder_center_x', state.cylinder_center_x))
    
    elif action.type == ActionType.SET_CYLINDER_CENTER_Y:
        return replace(state, cylinder_center_y=action.payload.get('cylinder_center_y', state.cylinder_center_y))
    
    elif action.type == ActionType.SET_OBSTACLE_POSITION:
        # Generic position update based on obstacle type
        obstacle_type = action.payload.get('obstacle_type', state.obstacle_type)
        x = action.payload.get('x')
        y = action.payload.get('y')
        
        if obstacle_type == 'naca_airfoil':
            return replace(state, naca_x=x if x is not None else state.naca_x, 
                                 naca_y=y if y is not None else state.naca_y)
        elif obstacle_type == 'cylinder':
            return replace(state, cylinder_center_x=x if x is not None else state.cylinder_center_x,
                                 cylinder_center_y=y if y is not None else state.cylinder_center_y)
        elif obstacle_type == 'cow':
            return replace(state, cow_x=x if x is not None else state.cow_x,
                                 cow_y=y if y is not None else state.cow_y)
        elif obstacle_type == 'three_cylinder_array':
            return replace(state, three_cylinder_x=x if x is not None else state.three_cylinder_x,
                                 three_cylinder_y=y if y is not None else state.three_cylinder_y)
        return state
    
    return state


def simulation_reducer(state: SimulationState, action: Action) -> SimulationState:
    """
    Reducer for simulation-related state changes.
    
    Args:
        state: Current simulation state
        action: Action to process
        
    Returns:
        New simulation state
    """
    if action.type == ActionType.SET_REYNOLDS_NUMBER:
        return replace(state, reynolds_number=action.payload.get('reynolds_number', state.reynolds_number))
    
    elif action.type == ActionType.SET_U_INF:
        return replace(state, u_inf=action.payload.get('u_inf', state.u_inf))
    
    elif action.type == ActionType.SET_NU:
        return replace(state, nu=action.payload.get('nu', state.nu))
    
    elif action.type == ActionType.SET_GRID_NX:
        return replace(state, grid_nx=action.payload.get('grid_nx', state.grid_nx))
    
    elif action.type == ActionType.SET_GRID_NY:
        return replace(state, grid_ny=action.payload.get('grid_ny', state.grid_ny))
    
    return state


def root_reducer(state: AppState, action: Action) -> AppState:
    """
    Root reducer that delegates to slice reducers.
    
    Args:
        state: Current application state
        action: Action to process
        
    Returns:
        New application state
    """
    return AppState(
        obstacle=obstacle_reducer(state.obstacle, action),
        simulation=simulation_reducer(state.simulation, action),
    )


# ============================================================================
# Store
# ============================================================================

class Store:
    """
    Redux-like store for managing application state.
    
    Provides:
    - Central state container
    - Action dispatch mechanism
    - Subscription system for state changes
    """
    
    def __init__(self, reducer: Callable, initial_state: AppState):
        """
        Initialize the store.
        
        Args:
            reducer: Root reducer function
            initial_state: Initial application state
        """
        self._reducer = reducer
        self._state = initial_state
        self._subscribers: list[Callable[[AppState], None]] = []
        self._middleware: list[Callable] = []
    
    def get_state(self) -> AppState:
        """Get the current state."""
        return self._state
    
    def dispatch(self, action: Action) -> None:
        """
        Dispatch an action to update state.
        
        Args:
            action: Action to dispatch
        """
        print(f"[STORE] Dispatching action: {action.type.value} with payload: {action.payload}")
        
        # Apply middleware (future enhancement)
        for middleware in self._middleware:
            action = middleware(self, action)
        
        # Apply reducer to get new state
        new_state = self._reducer(self._state, action)
        
        # Only update and notify if state actually changed
        if new_state != self._state:
            # Update state
            self._state = new_state
            
            # Notify subscribers
            self._notify()
    
    def subscribe(self, subscriber: Callable[[AppState], None]) -> Callable:
        """
        Subscribe to state changes.
        
        Args:
            subscriber: Callback function to call on state changes
            
        Returns:
            Unsubscribe function
        """
        self._subscribers.append(subscriber)
        
        def unsubscribe():
            """Remove the subscriber."""
            if subscriber in self._subscribers:
                self._subscribers.remove(subscriber)
        
        return unsubscribe
    
    def add_middleware(self, middleware: Callable) -> None:
        """
        Add middleware to the store.
        
        Args:
            middleware: Middleware function
        """
        self._middleware.append(middleware)
    
    def _notify(self) -> None:
        """Notify all subscribers of state change."""
        for subscriber in self._subscribers:
            try:
                subscriber(self._state)
            except Exception as e:
                print(f"[STORE] Error in subscriber: {e}")


# ============================================================================
# Middleware
# ============================================================================

def logging_middleware(store: 'Store', action: Action) -> Action:
    """
    Logging middleware that logs all actions dispatched to the store.
    
    Args:
        store: The store instance
        action: The action being dispatched
        
    Returns:
        The action (unchanged)
    """
    print(f"[MIDDLEWARE] Action: {action.type.value} | Payload: {action.payload}")
    return action


# ============================================================================
# Action Creators
# ============================================================================

def set_obstacle_type(obstacle_type: str) -> Action:
    """Action creator for setting obstacle type."""
    return Action(type=ActionType.SET_OBSTACLE_TYPE, payload={'obstacle_type': obstacle_type})


def set_naca_airfoil(naca_airfoil: str) -> Action:
    """Action creator for setting NACA airfoil."""
    return Action(type=ActionType.SET_NACA_AIRFOIL, payload={'naca_airfoil': naca_airfoil})


def set_naca_chord(naca_chord: float) -> Action:
    """Action creator for setting NACA chord."""
    return Action(type=ActionType.SET_NACA_CHORD, payload={'naca_chord': naca_chord})


def set_naca_angle(naca_angle: float) -> Action:
    """Action creator for setting NACA angle."""
    return Action(type=ActionType.SET_NACA_ANGLE, payload={'naca_angle': naca_angle})


def set_naca_x(naca_x: float) -> Action:
    """Action creator for setting NACA X position."""
    return Action(type=ActionType.SET_NACA_X, payload={'naca_x': naca_x})


def set_naca_y(naca_y: float) -> Action:
    """Action creator for setting NACA Y position."""
    return Action(type=ActionType.SET_NACA_Y, payload={'naca_y': naca_y})


def set_cylinder_radius(cylinder_radius: float) -> Action:
    """Action creator for setting cylinder radius."""
    return Action(type=ActionType.SET_CYLINDER_RADIUS, payload={'cylinder_radius': cylinder_radius})


def set_cylinder_center_x(cylinder_center_x: float) -> Action:
    """Action creator for setting cylinder center X."""
    return Action(type=ActionType.SET_CYLINDER_CENTER_X, payload={'cylinder_center_x': cylinder_center_x})


def set_cylinder_center_y(cylinder_center_y: float) -> Action:
    """Action creator for setting cylinder center Y."""
    return Action(type=ActionType.SET_CYLINDER_CENTER_Y, payload={'cylinder_center_y': cylinder_center_y})


# Position change actions (for live preview updates)
def set_obstacle_position(obstacle_type: str, x: float, y: float) -> Action:
    """Action creator for setting obstacle position based on obstacle type."""
    payload = {'obstacle_type': obstacle_type, 'x': x, 'y': y}
    return Action(type=ActionType.SET_OBSTACLE_POSITION, payload=payload)


def set_reynolds_number(reynolds_number: float) -> Action:
    """Action creator for setting Reynolds number."""
    return Action(type=ActionType.SET_REYNOLDS_NUMBER, payload={'reynolds_number': reynolds_number})


def set_u_inf(u_inf: float) -> Action:
    """Action creator for setting inlet velocity."""
    return Action(type=ActionType.SET_U_INF, payload={'u_inf': u_inf})


def set_nu(nu: float) -> Action:
    """Action creator for setting kinematic viscosity."""
    return Action(type=ActionType.SET_NU, payload={'nu': nu})


def set_grid_nx(grid_nx: int) -> Action:
    """Action creator for setting grid X resolution."""
    return Action(type=ActionType.SET_GRID_NX, payload={'grid_nx': grid_nx})


def set_grid_ny(grid_ny: int) -> Action:
    """Action creator for setting grid Y resolution."""
    return Action(type=ActionType.SET_GRID_NY, payload={'grid_ny': grid_ny})


# ============================================================================
# Global Store Instance
# ============================================================================

# Create global store instance
initial_state = AppState()
store = Store(root_reducer, initial_state)

# Add logging middleware
store.add_middleware(logging_middleware)
