<?php
// Terminate the Python script
$output = shell_exec('pkill -f "python drowsiness_detection.py"');
echo "Program stopped";
?>
