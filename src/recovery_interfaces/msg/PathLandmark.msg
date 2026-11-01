# PathLandmark message: lightweight landmark record used by the publisher /
# subscriber data-collection alternative. The service-based recording path
# stores the same information through RecordLandmark.srv, so this message is
# provided for interface completeness.

std_msgs/String msg       # trigger / label string used before processing
float64 time_taken        # recorded sampling period T (s)
float64 angle_yaw         # monitored yaw angle (rad)
