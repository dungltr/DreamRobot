# PathLandmark message (Figure 9 C in the paper).
#
# This is the lightweight three-variable message used by the publisher /
# subscriber data-collection alternative described in the paper
# (section 3.4.1). The service-based recording path used in this
# implementation stores the same information through RecordLandmark.srv, so
# this message is provided for interface completeness against Figure 9.

std_msgs/String msg       # trigger / label string used before processing
float64 time_taken        # recorded sampling period T (s)
float64 angle_yaw         # monitored yaw angle (rad)
