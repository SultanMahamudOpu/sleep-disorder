# Sleep Disorder Prediction Web App

এই web application টি Machine Learning ব্যবহার করে Sleep Disorder predict করে। আপনার স্বাস্থ্য এবং জীবনযাত্রার তথ্যের ভিত্তিতে এটি নির্ণয় করে আপনার Insomnia বা Sleep Apnea আছে কিনা।

## to run this app
cd "c:\Users\Sultan Mahamud Opu\Desktop\RU APP"; streamlit run streamlit_app.py

## Features

- 🔮 **Sleep Disorder Prediction**: ব্যক্তিগত তথ্যের ভিত্তিতে sleep disorder predict করে
- 📊 **Data Visualization**: Dataset এর বিভিন্ন visualization
- 💡 **Recommendations**: Prediction এর ভিত্তিতে health recommendations
- 🎨 **Beautiful UI**: Modern এবং user-friendly interface

## ব্যবহৃত Technology

- **Frontend**: Streamlit
- **Machine Learning**: Scikit-learn (Gradient Boosting Classifier)
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly
- **Language**: Python

## Installation এবং Setup

### 1. Repository Clone করুন
```bash
git clone <repository-url>
cd sleep-disorder-prediction
```

### 2. Virtual Environment তৈরি করুন (Optional)
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac
```

### 3. Required Packages Install করুন
```bash
pip install -r requirements.txt
```

### 4. Model Train করুন
```bash
python simple_train.py
```

### 5. Web App Run করুন
```bash
streamlit run streamlit_app.py
```

## Files এর বিবরণ

- `streamlit_app.py`: Main web application
- `simple_train.py`: Model training script
- `sleep.csv`: Dataset
- `requirements.txt`: Required Python packages
- `*.pkl`: Trained model এবং encoder files

## Model Performance

- **Algorithm**: Gradient Boosting Classifier
- **Accuracy**: 94.52%
- **Classes**: 
  - No Sleep Disorder
  - Insomnia
  - Sleep Apnea

## Dataset Features

Model নিম্নলিখিত features ব্যবহার করে:

1. **Personal Information**:
   - Gender
   - Age
   - Occupation

2. **Sleep Metrics**:
   - Sleep Duration
   - Quality of Sleep

3. **Health Metrics**:
   - BMI Category
   - Heart Rate
   - Blood Pressure (Systolic/Diastolic)

4. **Lifestyle**:
   - Physical Activity Level
   - Stress Level
   - Daily Steps

## ব্যবহারবিধি

1. **Prediction Page**: আপনার তথ্য input করুন এবং prediction দেখুন
2. **Data Visualization**: Dataset এর analysis এবং insights দেখুন
3. **About Page**: Application সম্পর্কে বিস্তারিত জানুন

## Important Disclaimer

⚠️ **এই tool শুধুমাত্র educational এবং informational উদ্দেশ্যে।** এটি professional medical advice, diagnosis, বা treatment এর বিকল্প নয়। Sleep-related যেকোনো সমস্যার জন্য qualified healthcare professional এর সাথে পরামর্শ করুন।

## Contribution

Contributions are welcome! Please feel free to submit issues বা pull requests।

## License

This project is licensed under the MIT License.

## Contact

কোনো প্রশ্ন থাকলে যোগাযোগ করুন।



