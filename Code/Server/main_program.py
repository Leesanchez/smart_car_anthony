import time
import threading
from line_follower import LineFollower
from maze_solver import MazeSolver
from infrared import Infrared
from ultrasonic import Ultrasonic
from motor import Ordinary_Car
from buzzer import Buzzer

class SmartCar:
    def __init__(self):
        # Initialize components
        self.infrared = Infrared()
        self.ultrasonic = Ultrasonic()
        self.motor = Ordinary_Car()
        self.buzzer = Buzzer()
        
        # Initialize modes
        self.line_follower = LineFollower()
        self.maze_solver = MazeSolver()
        
        # State variables
        self.current_mode = "line"  # "line" or "maze"
        self.is_running = False
        self.found_maze_entrance = False
        
        # Line detection threshold
        self.LINE_THRESHOLD = 1000  # Adjust based on your sensor
        
    def check_maze_entrance(self):
        """Check if we've reached the maze entrance."""
        # Get infrared readings
        left = self.infrared.read_one_infrared(1)
        middle = self.infrared.read_one_infrared(2)
        right = self.infrared.read_one_infrared(3)
        
        # If all sensors detect line (or no line), might be maze entrance
        if (left > self.LINE_THRESHOLD and 
            middle > self.LINE_THRESHOLD and 
            right > self.LINE_THRESHOLD):
            # Double check with ultrasonic
            distance = self.ultrasonic.get_distance()
            if distance < 30:  # If wall detected within 30cm
                return True
        return False
    
    def switch_to_maze_mode(self):
        """Switch from line following to maze solving."""
        print("Switching to maze mode...")
        self.motor.set_motor_model(0, 0, 0, 0)  # Stop
        time.sleep(1)
        self.buzzer.beep(0.5)  # Signal mode change
        self.current_mode = "maze"
        self.maze_solver.current_position = (0, 0)  # Reset position
        self.maze_solver.current_direction = 0  # Reset direction (North)
        
    def run(self):
        """Main run loop."""
        self.is_running = True
        print("Starting in line following mode...")
        
        try:
            while self.is_running:
                if self.current_mode == "line":
                    # Run line following
                    self.line_follower.follow_line()
                    
                    # Check for maze entrance
                    if self.check_maze_entrance():
                        self.switch_to_maze_mode()
                        
                else:  # maze mode
                    # Run maze solving
                    success = self.maze_solver.run_step()
                    if not success:
                        print("Maze solving step failed, trying to recover...")
                        time.sleep(0.5)
                        
                time.sleep(0.01)  # Small delay to prevent CPU overload
                
        except KeyboardInterrupt:
            print("\nProgram stopped by user")
        finally:
            self.cleanup()
            
    def cleanup(self):
        """Clean up resources."""
        self.is_running = False
        self.motor.set_motor_model(0, 0, 0, 0)
        print("Program terminated safely")

def main():
    car = SmartCar()
    car.run()

if __name__ == "__main__":
    main() 