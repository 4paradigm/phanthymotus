# Robot-frame direction: front 0, left positive (counter-clockwise, REP-103,
# same sign as vyaw), range (-180, 180].
# VALID carries an angle and a capture window. NO_EVENT clears live display.
# Windows use the producer's shared Unix capture clock in microseconds;
# zero means untrusted timing and must not be used for audio matching.
uint8 NO_EVENT=0
uint8 VALID=1
uint8 state
float32 angle_deg
int64 window_start_us
int64 window_end_us
int64 published_us
