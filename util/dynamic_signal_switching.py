def switch_signal(vehicle_counts):
    if not vehicle_counts:
        return "No lanes detected"
    return max(vehicle_counts, key=vehicle_counts.get)

def avg_signal_oc_time(vehicle_counts):
    if not vehicle_counts:
        return 0
    total = sum(vehicle_counts.values())
    lanes = len(vehicle_counts)
    return total / lanes