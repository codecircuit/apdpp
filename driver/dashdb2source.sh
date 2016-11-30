#!/bin/bash
echo "#ifndef MEKONG_BSP_DATABASE_H"
echo "#define MEKONG_BSP_DATABASE_H"
echo "namespace Mekong {"
echo "const char* bspAnalysisStr ="
cat $1 | sed s/^/\"/ | sed 's,$,\\n",' | sed 's,^"\\n"$,,'
echo ";"
echo "};"
echo "#endif"
