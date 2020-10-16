#!/bin/bash
CLAIM_ENV="/home/$USER/anaconda/envs/forecast"
if test -d "$CLAIM_ENV" ; then
    echo "environment exists"
else
    echo "environment doesn't exist"
fi

echo $CLAIM_ENV
test -d "$CLAIM_ENV"
