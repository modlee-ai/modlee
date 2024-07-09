#!/bin/sh
echo "Label,Text\n$(cat $1)" > $1
