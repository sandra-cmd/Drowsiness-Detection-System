# Driver Drowsiness Detection System  

## Introduction  
Driver fatigue is a significant cause of road accidents worldwide. This project introduces a **Driver Drowsiness Detection System** that uses image processing and machine learning to monitor driver behavior and detect signs of fatigue in real time. The system aims to enhance road safety by identifying early indicators of drowsiness and issuing timely alerts to prevent accidents.  

## Features  
- **Real-Time Detection**: Monitors driver facial expressions continuously.  
- **Image Processing**: Calculates metrics such as Eye Aspect Ratio (EAR), Mouth Opening Ratio (MOR), and Nose Length Ratio (NLR).  
- **High Accuracy**: Achieves 95.58% sensitivity and 100% specificity using Support Vector Machine (SVM).  
- **Alert Mechanism**: Issues audio alarms to alert drivers when drowsiness is detected.  
- **Portable and Cost-Effective**: Compatible with various vehicle types and designed for easy implementation.  

## Technologies Used  
- **Programming Language**: Python  
- **Libraries and Frameworks**:  
  - OpenCV for image processing  
  - Dlib for facial landmark detection  
  - TensorFlow for machine learning  
  - NumPy and SciPy for numerical computations  

## System Architecture  
1. **Video Capture**: A webcam captures the driver’s facial expressions in real time.  
2. **Facial Landmark Detection**: Detects key facial points, such as eyes and mouth.  
3. **Metric Calculation**: EAR, MOR, and NLR are calculated to assess fatigue.  
4. **Machine Learning**: SVM classifies drowsiness based on the metrics.  
5. **Alert Trigger**: When fatigue is detected, an alarm is triggered to alert the driver.  

## Installation  
1. Clone the repository:  
   ```bash  
   git clone https://github.com/your-username/driver-drowsiness-detection.git  
   ```  
2. Navigate to the project directory:  
   ```bash  
   cd driver-drowsiness-detection  
   ```  
3. Install dependencies:  
   ```bash  
   pip install -r requirements.txt  
   ```  
4. Run the application:  
   ```bash  
   python main.py  
   ```  

## Usage  
- Ensure a webcam is connected.  
- Run the script to start real-time monitoring.  
- The system will issue an alert if drowsiness is detected.  

## Results  
- **Accuracy**: 95.58% sensitivity and 100% specificity.  
- The system detects prolonged eye closure, yawning, and head tilting as signs of drowsiness.  

## Future Scope  
- Integration with vehicle systems for automatic interventions.  
- Enhanced detection using additional physiological data.  
- Mobile app integration for wider accessibility.  

## License  
This project is licensed under the [MIT License](LICENSE).  

## Acknowledgments  
Special thanks to the team and contributors who made this project possible.  

---  
Feel free to contribute to this repository by submitting pull requests or reporting issues!
