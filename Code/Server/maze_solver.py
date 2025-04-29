import time
import math
import threading
import numpy as np
from motor import *
from infrared import Infrared
from ultrasonic import Ultrasonic
from buzzer import Buzzer
from servo import Servo
from motor import Ordinary_Car
from queue import PriorityQueue
import json
from enum import IntEnum
from collections import deque

# Direction enum for consistent direction handling
class Dir(IntEnum):
    STRAIGHT = 0
    LEFT = 1
    RIGHT = 2
    BACK = -1

# Initialize sensors
infrared = Infrared()
ultrasonic = Ultrasonic()
buzzer = Buzzer()
servo = Servo()
PWM = Ordinary_Car()

class MazeSolver:
    def __init__(self):
        self.pwm = PWM
        
        # Constants for movement
        self.BASE_SPEED = 2000
        self.TURN_SPEED = 2500
        self.TURN_DURATION_90 = 0.65  # Increased from 0.55 for more accurate turns
        self.CELL_DURATION = 0.2  # seconds to travel one grid cell
        
        # Maze solving variables
        self.grid_size = 20  # cm per grid cell
        self.current_position = (0, 0)  # (x, y) in grid coordinates
        self.current_direction = 0  # 0: North, 1: East, 2: South, 3: West
        self.maze_map = {}  # Dictionary to store discovered maze cells: {(x,y): [north_wall, east_wall, south_wall, west_wall]}
        self.visited_cells = set()  # Set of visited cells
        self.path = []  # Current A* path
        self.target_position = None  # Target position for pathfinding
        self.in_maze = False  # Whether we're in the maze or still following the line
        
        # Maze dimensions
        self.min_x = 0
        self.max_x = 0
        self.min_y = 0
        self.max_y = 0
        
        # Stuck detection
        self.last_positions = deque(maxlen=8)
        
        # Servo scanning parameters
        self.SCAN_ANGLES = [15, 45, 75, 90, 105, 135, 165] # 7-point scan
        self.SERVO_MOVE_TIME = 0.5  # Increased pause time for servo to settle
        self.READING_PAUSE = 0.05  # Small pause after reading
        
        # Angle definitions for readability
        self.ANG_LEFT_FAR = 15
        self.ANG_LEFT_MID = 45
        self.ANG_LEFT_NEAR = 75
        self.ANG_FRONT = 90
        self.ANG_RIGHT_NEAR = 105
        self.ANG_RIGHT_MID = 135
        self.ANG_RIGHT_FAR = 165
        self.scan_results = {}  # Store scan results {angle: distance}
        self.observations = []
        
        # Wall detection threshold - ADJUSTED
        self.FRONT_CLEAR_CM = 45     # start the turn sooner
        self.SIDE_CLEAR_CM  = 35     # sides may be slightly closer
        self.MAP_WALL_THRESHOLD_CM = 30  # Threshold for marking walls in the map
        
        # For simulation mode (no hardware)
        self.simulation_mode = False

    def set_simulation_mode(self, mode):
        """Set simulation mode (True for no hardware, False for hardware)"""
        self.simulation_mode = mode
    
    def median_distance(self):
        """Get median distance reading, ignoring zeros"""
        samples = [d for d in (ultrasonic.get_distance() for _ in range(5)) if d > 0]
        if not samples:
            return 1  # Treat as blocked
        samples.sort()
        return samples[len(samples)//2]
        
    def scan_environment(self):
        """Scan the environment using the servo-mounted ultrasonic sensor."""
        self.scan_results = {}

        if self.simulation_mode:
            # Simulated scan results
            self.scan_results = {
                15: 50, 45: 60, 75: 70,  # Left side
                90: 25,  # Front (blocked)
                105: 70, 135: 60, 165: 50  # Right side
            }
            time.sleep(0.5)
            return self.scan_results

        # Start scan sequence
        servo.set_servo_pwm('0', self.SCAN_ANGLES[0])
        time.sleep(0.3)

        for angle in self.SCAN_ANGLES:
            servo.set_servo_pwm('0', angle)
            time.sleep(self.SERVO_MOVE_TIME)
            distance = self.median_distance()
            self.scan_results[angle] = distance
            time.sleep(self.READING_PAUSE)

        # Reset to center
        servo.set_servo_pwm('0', self.ANG_FRONT)
        time.sleep(0.2)
        return self.scan_results
    
    def get_direction(self, scan):
        """Decide direction based on scan results."""
        fwd_dist = scan.get(self.ANG_FRONT, 0)
        if fwd_dist > self.FRONT_CLEAR_CM:
            return Dir.STRAIGHT

        left_readings = [scan.get(a, 0) for a in self.SCAN_ANGLES if a < self.ANG_FRONT]
        right_readings = [scan.get(a, 0) for a in self.SCAN_ANGLES if a > self.ANG_FRONT]

        max_left = max(left_readings) if left_readings else 0
        max_right = max(right_readings) if right_readings else 0

        left_clear = max_left > self.SIDE_CLEAR_CM
        right_clear = max_right > self.SIDE_CLEAR_CM

        if left_clear or right_clear:
            return Dir.LEFT if max_left >= max_right else Dir.RIGHT

        return Dir.BACK
    
    def update_maze_map(self):
        """Update maze map based on current position and scan results."""
        x, y = self.current_position
        self.visited_cells.add(self.current_position)

        # Update boundaries
        self.min_x = min(self.min_x, x)
        self.max_x = max(self.max_x, x)
        self.min_y = min(self.min_y, y)
        self.max_y = max(self.max_y, y)

        if self.current_position not in self.maze_map:
            self.maze_map[self.current_position] = [False, False, False, False]

        scan = self.scan_results

        # Determine walls
        front_dist = scan.get(self.ANG_FRONT, 0)
        front_wall = front_dist < self.MAP_WALL_THRESHOLD_CM

        left_readings = [scan.get(a, 0) for a in self.SCAN_ANGLES if a < self.ANG_FRONT and a in scan]
        min_left = min(left_readings) if left_readings else 999
        left_wall = min_left < self.MAP_WALL_THRESHOLD_CM

        right_readings = [scan.get(a, 0) for a in self.SCAN_ANGLES if a > self.ANG_FRONT and a in scan]
        min_right = min(right_readings) if right_readings else 999
        right_wall = min_right < self.MAP_WALL_THRESHOLD_CM

        # Map relative walls to absolute directions
        relative_walls = {
            0: front_wall,   # Front
            1: right_wall,   # Right
            3: left_wall     # Left
        }

        # Convert relative walls to absolute walls based on current direction
        for rel_dir, wall_present in relative_walls.items():
            abs_dir = (rel_dir + self.current_direction) % 4
            self.maze_map[self.current_position][abs_dir] = wall_present

            # Update adjacent cell's corresponding wall
            dx = dy = 0
            if abs_dir == 0:   # North
                dy = 1
            elif abs_dir == 1: # East
                dx = 1
            elif abs_dir == 2: # South
                dy = -1
            elif abs_dir == 3: # West
                dx = -1

            adjacent_pos = (x + dx, y + dy)
            if wall_present and adjacent_pos not in self.maze_map:
                self.maze_map[adjacent_pos] = [False, False, False, False]
                self.maze_map[adjacent_pos][(abs_dir + 2) % 4] = True

    def a_star_pathfinding(self, start, goal):
        """A* pathfinding algorithm."""
        frontier = PriorityQueue()
        frontier.put((0, start))
        came_from = {start: None}
        cost_so_far = {start: 0}

        while not frontier.empty():
            current = frontier.get()[1]

            if current == goal:
                break

            # Check all four directions
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                next_pos = (current[0] + dx, current[1] + dy)

                # Skip if wall blocks the way
                dir_index = {(0, 1): 0, (1, 0): 1, (0, -1): 2, (-1, 0): 3}[(dx, dy)]
                if current in self.maze_map and self.maze_map[current][dir_index]:
                    continue

                new_cost = cost_so_far[current] + 1
                if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                    cost_so_far[next_pos] = new_cost
                    priority = new_cost + self.heuristic(goal, next_pos)
                    frontier.put((priority, next_pos))
                    came_from[next_pos] = current

        # Reconstruct path
        if goal not in came_from:
            return None

        path = []
        current = goal
        while current is not None:
            path.append(current)
            current = came_from[current]
        path.reverse()
        return path

    def heuristic(self, a, b):
        """Manhattan distance heuristic for A*."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def get_next_exploration_target(self):
        """Find the nearest unexplored cell."""
        if not self.visited_cells:
            return (0, 0)

        # Get all cells adjacent to visited cells
        frontier_cells = set()
        for x, y in self.visited_cells:
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                new_pos = (x + dx, y + dy)
                if new_pos not in self.visited_cells:
                    frontier_cells.add(new_pos)

        # Find the nearest frontier cell
        if frontier_cells:
            return min(frontier_cells, key=lambda pos: self.heuristic(self.current_position, pos))
        return None

    def go_straight(self, cells=1):
        """Move forward specified number of cells."""
        if self.simulation_mode:
            time.sleep(cells * self.CELL_DURATION)
            return

        self.pwm.set_motor_model(self.BASE_SPEED, self.BASE_SPEED, 
                                self.BASE_SPEED, self.BASE_SPEED)
        time.sleep(cells * self.CELL_DURATION)
        self.pwm.set_motor_model(0, 0, 0, 0)

    def turn(self, left=True):
        """Turn left or right."""
        if self.simulation_mode:
            time.sleep(self.TURN_DURATION_90)
            return

        if left:
            self.pwm.set_motor_model(-self.TURN_SPEED, -self.TURN_SPEED,
                                    self.TURN_SPEED, self.TURN_SPEED)
        else:
            self.pwm.set_motor_model(self.TURN_SPEED, self.TURN_SPEED,
                                    -self.TURN_SPEED, -self.TURN_SPEED)
        time.sleep(self.TURN_DURATION_90)
        self.pwm.set_motor_model(0, 0, 0, 0)

    def safety_check(self):
        """Check if movement is safe."""
        front_dist = ultrasonic.get_distance()
        if front_dist < 10:  # Emergency stop if too close
            self.pwm.set_motor_model(0, 0, 0, 0)
            return False
        return True

    def move(self, direction):
        """Execute movement in specified direction."""
        if not self.safety_check():
            return False

        # Calculate required turns
        turns_needed = 0
        if direction != Dir.STRAIGHT:
            current_rel_dir = direction.value if direction != Dir.BACK else 2
            turns_needed = (current_rel_dir - self.current_direction) % 4

        # Execute turns
        for _ in range(turns_needed):
            self.turn(left=True)
            self.current_direction = (self.current_direction + 1) % 4
            time.sleep(0.1)

        # Move forward
        if direction != Dir.BACK:
            self.go_straight()
            self._update_position_after_straight()
        else:
            # For backward movement, turn 180° and go forward
            self.turn(left=True)
            self.turn(left=True)
            self.current_direction = (self.current_direction + 2) % 4
            self.go_straight()
            self._update_position_after_straight()

        return True

    def _update_position_after_straight(self):
        """Update position after moving forward."""
        dx = dy = 0
        if self.current_direction == 0:    # North
            dy = 1
        elif self.current_direction == 1:  # East
            dx = 1
        elif self.current_direction == 2:  # South
            dy = -1
        elif self.current_direction == 3:  # West
            dx = -1

        self.current_position = (
            self.current_position[0] + dx,
            self.current_position[1] + dy
        )
        self.last_positions.append(self.current_position)

    def follow_path(self, path):
        """Follow a given path."""
        if not path or len(path) < 2:
            return False

        for i in range(1, len(path)):
            current = path[i-1]
            next_pos = path[i]
            
            # Calculate direction to next position
            dx = next_pos[0] - current[0]
            dy = next_pos[1] - current[1]
            
            # Convert to absolute direction (0=North, 1=East, 2=South, 3=West)
            target_direction = {
                (0, 1): 0,   # North
                (1, 0): 1,   # East
                (0, -1): 2,  # South
                (-1, 0): 3   # West
            }[(dx, dy)]
            
            # Calculate turns needed
            turns = (target_direction - self.current_direction) % 4
            if turns == 1:
                self.turn(left=False)  # Turn right
            elif turns == 2:
                self.turn(left=True)
                self.turn(left=True)
            elif turns == 3:
                self.turn(left=True)   # Turn left
            
            self.current_direction = target_direction
            self.go_straight()
            self._update_position_after_straight()
            
        return True

    def run_step(self):
        """Execute one step of maze exploration."""
        # Scan surroundings
        scan = self.scan_environment()
        
        # Update map
        self.update_maze_map()
        
        # Get movement direction
        direction = self.get_direction(scan)
        
        # Execute movement
        success = self.move(direction)
        
        return success

    def run_exploration(self, steps=10):
        """Run maze exploration for specified number of steps."""
        for _ in range(steps):
            if not self.run_step():
                break
            time.sleep(0.1)

    def run_to_target(self, target_position):
        """Navigate to a target position using A*."""
        path = self.a_star_pathfinding(self.current_position, target_position)
        if path:
            return self.follow_path(path)
        return False

    def save_maze_map(self, filename='maze_map.json'):
        """Save the maze map to a JSON file."""
        map_data = {
            'maze_map': {str(k): v for k, v in self.maze_map.items()},
            'visited_cells': list(self.visited_cells),
            'current_position': self.current_position,
            'current_direction': self.current_direction,
            'dimensions': {
                'min_x': self.min_x,
                'max_x': self.max_x,
                'min_y': self.min_y,
                'max_y': self.max_y
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(map_data, f, indent=2)

    def load_maze_map(self, filename='maze_map.json'):
        """Load a maze map from a JSON file."""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
                
            self.maze_map = {eval(k): v for k, v in data['maze_map'].items()}
            self.visited_cells = set(tuple(x) for x in data['visited_cells'])
            self.current_position = tuple(data['current_position'])
            self.current_direction = data['current_direction']
            
            dims = data['dimensions']
            self.min_x = dims['min_x']
            self.max_x = dims['max_x']
            self.min_y = dims['min_y']
            self.max_y = dims['max_y']
            
            return True
        except Exception as e:
            print(f"Error loading maze map: {e}")
            return False

def main():
    solver = None
    try:
        solver = MazeSolver()
        solver.explore_maze()
    except KeyboardInterrupt:
        print("\nProgram stopped by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if solver:
            solver.cleanup()

if __name__ == "__main__":
    main()



