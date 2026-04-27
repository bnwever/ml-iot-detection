1. General Event Types (_log_type and service) The JSON dataset combines multiple Zeek logs. You can look at the _log_type field to see the category of the event:

conn: General TCP/UDP/ICMP connections.
dns, http, dhcp, ssl, ntp: Specific application-layer transaction events.
weird: Anomalous packet events (e.g., malformed packets, protocol violations).
If the record is a connection ("_log_type": "conn"), the service field will also tell you if Zeek identified the application protocol (e.g., "service": "http").

2. Packet-Level Events (history and conn_state) If you want to know the specific packet flags exchanged during a connection, you can look at the history field in the conn records. Zeek uses a string of characters to represent the sequence of packet events sent by the originator (uppercase) and the responder (lowercase):

S / s: SYN (Connection attempt)
h: SYN+ACK (Connection response)
A / a: ACK (Acknowledgment)
D / d: Data payload transferred
F / f: FIN (Normal connection teardown)
R / r: RST (Connection reset)
(For example, a history of "ShADadfF" means the connection saw a SYN, SYN+ACK, ACK, Data from both sides, and a normal FIN teardown).

The conn_state field also summarizes the final result of these packets (e.g., "SF" means a normal established and terminated connection, while "S0" means a connection attempt was made but no reply was seen).
