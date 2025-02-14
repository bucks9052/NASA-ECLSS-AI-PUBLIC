import random
import time
import csv
from datetime import datetime

# Subsystems in the ECLSS
subsystems = ['Oxygen Generator', 'CO2 Scrubber', 'Airlock', 'Water Processor']

# Log levels
alarm_levels = ['Normal', 'Watch', 'Minor Fault', 'Emergency']

def generate_log_entry():
    subsystem = random.choice(subsystems)
    # Random sensor value (simulate fault with 10% probability of out-of-range values)
    sensor_value = random.uniform(50, 150) if random.random() > 0.1 else random.uniform(150, 300)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Alarm levels based on the sensor value
    if sensor_value <= 100:
        alarm_level = 'Normal'
    elif 100 < sensor_value <= 125:
        alarm_level = 'Watch'
    elif 125 < sensor_value <= 150:
        alarm_level = 'Minor Fault'
    else:
        alarm_level = 'Emergency'

    return [subsystem, sensor_value, timestamp, alarm_level]

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
