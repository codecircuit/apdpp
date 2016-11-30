#!/bin/bash
echo "#ifndef MEKONG_USER_CONFIG_H"
echo "#define MEKONG_USER_CONFIG_H"
CONTENT=$(cat $1 | sed 's/#[^$]*//' | sed 's/^\([a-zA-Z]\)/#define \1/' | sed s/=//)
printf "$CONTENT\n"
echo "#endif"
