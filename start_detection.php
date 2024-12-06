<?php
// Execute the Python script to start the program
$output = shell_exec('python drowsiness_detection.py > /dev/null 2>&1 &');
echo "Program started";
?>
