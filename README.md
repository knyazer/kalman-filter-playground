# Simple benchmark for 6- and 4- state length Kalman Filters

This repository contains some a Proof Of Concept, that shows that shows that using K4 for ball but K6 for robots makes sense. Also, it outlines the importance of the careful tuning of Kalman Filter.

### Usage
Run ```python3 main.py $i```, where ```$i``` is 0, 1, 2 or 3 corresponding to different environments.
During excecution, press:
Esc - to exit
Space - to pause
k or l - to move forward in time
h or j - to move backward in time
1,2,3,4 - to switch visibility of a particular trace
