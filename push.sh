#!/bin/bash

# Check if a commit message was provided
if [ -z "$1" ]
then
  echo "Error: Please provide a commit message."
  echo "Usage: ./push.sh 'My commit message'"
  exit 1
fi

# Run the git commands
git add .
git commit -m "$1"
git push

echo "Successfully pushed to GitHub!"
