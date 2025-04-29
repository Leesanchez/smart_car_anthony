import time
from motor import *
from infrared import Infrared
from ultrasonic import Ultrasonic
from buzzer import Buzzer
from servo import Servo
import math

# Initialize sensors
infrared = Infrared()
ultrasonic = Ultrasonic()
buzzer = Buzzer()
servo = Servo()

class MazeSolver:
    def __init__(self):
        self.pwm = PCA9685(0x40, debug=True)
        self.pwm.set_pwm_freq(50)
        
        # Maze solving variables
        self.grid_size = 20  # cm per grid cell
        self.current_position = (0, 0)  # (x, y) in grid coordinates
        self.current_direction = 0  # 0: North, 1: East, 2: South, 3: West
        self.maze_map = {}  # Stores discovered walls
        self.path = []  # Current A* path
        self.in_maze = False  # Whether we're in the maze or still following the line
        
        # Servo scanning parameters
        self.scan_angles = [50, 70, 90, 110, 130]  # Servo angles to scan
        self.scan_results = {}  # Store scan results
        self.current_scan_index = 0

    def duty_range(self, duty1, duty2, duty3, duty4):
        if duty1 > 4095:
            duty1 = 4095
        elif duty1 < -4095:
            duty1 = -4095        
        if duty2 > 4095:
            duty2 = 4095
        elif duty2 < -4095:
            duty2 = -4095  
        if duty3 > 4095:
            duty3 = 4095
        elif duty3 < -4095:
            duty3 = -4095
        if duty4 > 4095:
            duty4 = 4095
        elif duty4 < -4095:
            duty4 = -4095
        return duty1, duty2, duty3, duty4

    def set_motor_model(self, duty1, duty2, duty3, duty4):
        duty1, duty2, duty3, duty4 = self.duty_range(duty1, duty2, duty3, duty4)
        
        # Left upper wheel
        if duty1 > 0:
            self.pwm.set_motor_pwm(0, 0)
            self.pwm.set_motor_pwm(1, duty1)
        elif duty1 < 0:
            self.pwm.set_motor_pwm(1, 0)
            self.pwm.set_motor_pwm(0, abs(duty1))
        else:
            self.pwm.set_motor_pwm(0, 4095)
            self.pwm.set_motor_pwm(1, 4095)
            
        # Left lower wheel
        if duty2 > 0:
            self.pwm.set_motor_pwm(3, 0)
            self.pwm.set_motor_pwm(2, duty2)
        elif duty2 < 0:
            self.pwm.set_motor_pwm(2, 0)
            self.pwm.set_motor_pwm(3, abs(duty2))
        else:
            self.pwm.set_motor_pwm(2, 4095)
            self.pwm.set_motor_pwm(3, 4095)
            
        # Right upper wheel
        if duty3 > 0:
            self.pwm.set_motor_pwm(6, 0)
            self.pwm.set_motor_pwm(7, duty3)
        elif duty3 < 0:
            self.pwm.set_motor_pwm(7, 0)
            self.pwm.set_motor_pwm(6, abs(duty3))
        else:
            self.pwm.set_motor_pwm(6, 4095)
            self.pwm.set_motor_pwm(7, 4095)
            
        # Right lower wheel
        if duty4 > 0:
            self.pwm.set_motor_pwm(4, 0)
            self.pwm.set_motor_pwm(5, duty4)
        elif duty4 < 0:
            self.pwm.set_motor_pwm(5, 0)
            self.pwm.set_motor_pwm(4, abs(duty4))
        else:
            self.pwm.set_motor_pwm(4, 4095)
            self.pwm.set_motor_pwm(5, 4095)

    def manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def get_neighbors(self, pos):
        x, y = pos
        neighbors = []
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            new_pos = (x + dx, y + dy)
            if new_pos not in self.maze_map.get(pos, []):
                neighbors.append(new_pos)
        return neighbors

    def a_star_search(self, start, goal):
        frontier = [(0, start)]
        came_from = {start: None}
        g_score = {start: 0}
        f_score = {start: self.manhattan_distance(start, goal)}

        while frontier:
            current = frontier.pop(0)[1]
            if current == goal:
                path = []
                while current is not None:
                    path.append(current)
                    current = came_from[current]
                return path[::-1]

            for neighbor in self.get_neighbors(current):
                tentative_g_score = g_score[current] + 1
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.manhattan_distance(neighbor, goal)
                    if neighbor not in [i[1] for i in frontier]:
                        frontier.append((f_score[neighbor], neighbor))
            frontier.sort()

        return None

    def scan_environment(self):
        """Scan the environment using the servo-mounted ultrasonic sensor"""
        self.scan_results = {}  # Clear previous results
        
        for angle in self.scan_angles:
            # Move servo to scan angle
            servo.set_servo_pwm('0', angle)
            time.sleep(0.2)  # Wait for servo to move
            
            # Take multiple measurements and average them
            measurements = []
            for _ in range(3):  # Take 3 measurements
                distance = ultrasonic.get_distance()
                if distance is not None:
                    measurements.append(distance)
                time.sleep(0.1)
            
            if measurements:
                avg_distance = sum(measurements) / len(measurements)
                self.scan_results[angle] = avg_distance
                print(f"Scan at {angle}°: {avg_distance:.1f}cm")
            
            time.sleep(0.1)  # Small delay between scans
        
        # Return servo to center position
        servo.set_servo_pwm('0', 90)
        time.sleep(0.2)

    def update_maze_map(self):
        """Update the maze map based on ultrasonic scan results"""
        # Scan the environment
        self.scan_environment()
        
        # Convert scan angles to relative directions
        angle_to_direction = {
            50: 'left',
            70: 'left-front',
            90: 'front',
            110: 'right-front',
            130: 'right'
        }
        
        # Update walls based on scan results
        for angle, distance in self.scan_results.items():
            if distance is not None and distance < 30:  # Wall detected
                direction = angle_to_direction[angle]
                if direction == 'left':
                    wall_pos = (self.current_position[0] - 1, self.current_position[1])
                elif direction == 'right':
                    wall_pos = (self.current_position[0] + 1, self.current_position[1])
                elif direction == 'front':
                    wall_pos = (self.current_position[0], self.current_position[1] + 1)
                elif direction == 'left-front':
                    wall_pos = (self.current_position[0] - 1, self.current_position[1] + 1)
                elif direction == 'right-front':
                    wall_pos = (self.current_position[0] + 1, self.current_position[1] + 1)
                
                if self.current_position not in self.maze_map:
                    self.maze_map[self.current_position] = []
                self.maze_map[self.current_position].append(wall_pos)

    def run(self):
        try:
            while True:
                # First check for obstacles
                distance = ultrasonic.get_distance()
                if distance is not None and distance < 30:
                    self.set_motor_model(0, 0, 0, 0)
                    buzzer.set_state(1)  # Turn buzzer on
                    time.sleep(0.2)
                    buzzer.set_state(0)  # Turn buzzer off
                    self.set_motor_model(-1500, -1500, 1500, 1500)
                    time.sleep(0.5)
                    continue

                # Read infrared sensors using the Infrared class
                ir1_value = infrared.read_one_infrared(1)
                ir2_value = infrared.read_one_infrared(2)
                ir3_value = infrared.read_one_infrared(3)

                # If we're still following the line (middle sensor detects line)
                if ir2_value == 1 and not self.in_maze:
                    if ir1_value != 1 and ir2_value == 1 and ir3_value != 1:
                        # Only middle sensor detects line - go straight
                        self.set_motor_model(800, 800, 800, 800)
                    elif ir1_value != 1 and ir2_value != 1 and ir3_value == 1:
                        # Right sensor detects line - turn right
                        self.set_motor_model(-1500, -1500, 2500, 2500)
                    elif ir1_value == 1 and ir2_value != 1 and ir3_value != 1:
                        # Left sensor detects line - turn left
                        self.set_motor_model(2500, 2500, -1500, -1500)
                    elif ir1_value == 1 and ir2_value == 1 and ir3_value != 1:
                        # Left and middle sensors detect line - sharp left turn
                        self.set_motor_model(4000, 4000, -2000, -2000)
                    elif ir1_value != 1 and ir2_value == 1 and ir3_value == 1:
                        # Middle and right sensors detect line - sharp right turn
                        self.set_motor_model(-2000, -2000, 4000, 4000)
                    elif ir1_value == 1 and ir2_value == 1 and ir3_value == 1:
                        # All sensors detect line - intersection
                        self.set_motor_model(800, 800, 800, 800)
                elif not self.in_maze:
                    # If we lose the line, try to find it by turning right
                    self.set_motor_model(-1500, -1500, 1500, 1500)
                    time.sleep(0.5)
                else:
                    # We've entered the maze
                    self.in_maze = True
                    self.update_maze_map()

                    if not self.path:
                        goal_position = (5, 5)  # Example goal
                        self.path = self.a_star_search(self.current_position, goal_position)
                    
                    if self.path:
                        next_pos = self.path[0]
                        dx = next_pos[0] - self.current_position[0]
                        dy = next_pos[1] - self.current_position[1]
                        
                        required_direction = 0
                        if dx == 1: required_direction = 1
                        elif dx == -1: required_direction = 3
                        elif dy == 1: required_direction = 0
                        elif dy == -1: required_direction = 2
                        
                        turn_diff = (required_direction - self.current_direction) % 4
                        if turn_diff == 1:
                            self.set_motor_model(-1500, -1500, 1500, 1500)
                            time.sleep(0.5)
                        elif turn_diff == 3:
                            self.set_motor_model(1500, 1500, -1500, -1500)
                            time.sleep(0.5)
                        elif turn_diff == 2:
                            self.set_motor_model(-1500, -1500, 1500, 1500)
                            time.sleep(1)
                        
                        self.current_direction = required_direction
                        self.set_motor_model(800, 800, 800, 800)
                        time.sleep(1)
                        self.current_position = next_pos
                        self.path.pop(0)
                    else:
                        self.set_motor_model(-1500, -1500, 1500, 1500)
                        time.sleep(0.5)

        except KeyboardInterrupt:
            # Reset servo to center position
            servo.set_servo_pwm('0', 90)
            # Close the infrared sensor
            infrared.close()
            self.set_motor_model(0, 0, 0, 0)
            self.pwm.close()

if __name__ == '__main__':
    print('Program is starting...')
    maze_solver = MazeSolver()
    maze_solver.run() 