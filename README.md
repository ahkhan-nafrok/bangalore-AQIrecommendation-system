### 🏙️ Bangalore Recommendation System/

A full-stack web application that uses machine learning to recommend areas in Bangalore based on user preferences.
Built with React (frontend) and Flask (backend).


## 🚀 How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/ahkhan-nafrok/bangalore-AQIrecommendation-system.git
cd bangalore-AQIrecommendation-system


```
### 2. Backend Setup (Flask API)
```cd backend
python -m venv venv
source venv/bin/activate          # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

▶️ Run the Flask Server
```
python app.py
```

### 3. Frontend Setup (React App)
📌 Requirements: Node.js (v14 or newer)
```
cd ../frontend
npm install
npm start
```


🔄 Connecting Frontend & Backend
Ensure the Flask server (localhost:5000) is running before starting the React frontend.

The React frontend communicates with the backend using REST API calls to get recommended areas.


## 💡 Features
🎯 Input user preferences via interactive form

🧠 Backend ML model suggests best-fit areas

⚡ Fast response with pre-trained model

💅 Smooth, responsive UI with loading animation

🧠 ML Model Details
Trained using: Dataset in data/bangalore_areas.csv

Saved model: models/trained_model.pkl

Logic handled in: ml_model.py

The model analyzes various features and returns area recommendations based on user preferences.


## 🛠️ Technologies Used
``` Frontend	- Backend	ML Model
 React.js	Flask (Python)	scikit-learn
 HTML, CSS	REST API	pandas
 JavaScript	JSON Handling	pickle
```
