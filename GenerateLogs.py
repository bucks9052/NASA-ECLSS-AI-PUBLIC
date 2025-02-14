import random
import time
import csv
from datetime import datetime

# Subsystems in the ECLSS
subsystems = ['Oxygen Generator', 'CO2 Scrubber', 'Airlock', 'Water Processor']

# Log levels
alarm_levels = ['Normal', 'Watch', 'Minor Fault', 'Emergency']

# Initialize sensor history for rolling averages
sensor_history = {subsystem: random.uniform(80, 120) for subsystem in subsystems}

def generate_log_entry():
    subsystem = random.choice(subsystems)
    
    # Introduce rolling average: New value depends on previous ones
    previous_value = sensor_history[subsystem]
    drift = random.uniform(-5, 5)  # Small normal fluctuations
    sensor_value = previous_value + drift  

    # Simulate occasional spikes (10% probability)
    if random.random() < 0.1:
        sensor_value *= random.choice([1.5, 2])  # Sudden spike
    elif random.random() < 0.05:
        sensor_value *= 0.5  # Sudden drop

    # Clip values to realistic bounds
    sensor_value = max(50, min(sensor_value, 300))

    # Update rolling average (simple moving average)
    sensor_history[subsystem] = (sensor_history[subsystem] * 0.9) + (sensor_value * 0.1)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Assign Alarm Levels
    if sensor_value <= 100:
        alarm_level = 'Normal'
    elif 100 < sensor_value <= 125:
        alarm_level = 'Watch'
    elif 125 < sensor_value <= 150:
        alarm_level = 'Minor Fault'
    else:
        alarm_level = 'Emergency'

    return [subsystem, round(sensor_value, 2), timestamp, alarm_level]

# Generate logs
def generate_logs(num_entries=100):
    with open('eclss_logs.csv', 'w', newline='') as csvfile:
        log_writer = csv.writer(csvfile)
        log_writer.writerow(['Subsystem', 'SensorValue', 'Timestamp', 'AlarmLevel'])
        for _ in range(num_entries):
            log_entry = generate_log_entry()
            log_writer.writerow(log_entry)

if __name__ == "__main__":
    generate_logs(1000)  # Generates 1000 log entries
