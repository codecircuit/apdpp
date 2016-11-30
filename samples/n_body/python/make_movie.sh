#!/bin/bash
avconv -i ./pic/picture%04d.png -b 15000k -flags mv4 -r 25 -vf setpts=2*PTS ./movie.mp4
