import React, { useState, useEffect } from 'react';
import axios from 'axios';
import UserPreferences from './components/UserPreferences';
import Recommendations from './components/Recommendations';
import LoadingSpinner from './components/LoadingSpinner';
import './App.css';

const API_BASE_URL = 'http://localhost:5000/api';

function App() {
  const [loading, setLoading] = useState(false);
  const [recommendations, setRecommendations] = useState(null);
  const [error, setError] = useState(null);
  const [systemReady, setSystemReady] = useState(false);
  const [datasetInfo, setDatasetInfo] = useState(null);
  const [featureImportance, setFeatureImportance] = useState(null);

  // Check if backend is ready
  useEffect(() => {
    const checkSystemHealth = async () => {
      try {
        const response = await axios.get(`${API_BASE_URL}/health`);
        setSystemReady(response.data.ml_system_ready);
        
        if (response.data.ml_system_ready) {
          // Get dataset info
          const datasetResponse = await axios.get(`${API_BASE_URL}/dataset-info`);
          setDatasetInfo(datasetResponse.data);
          
          // Get feature importance
          const featureResponse = await axios.get(`${API_BASE_URL}/feature-importance`);
          setFeatureImportance(featureResponse.data);
        }
      } catch (err) {
        console.error('Backend connection failed:', err);
        setError('Backend service is not available. Please ensure the Flask server is running.');
      }
    };

    checkSystemHealth();
  }, []);

  const handleGetRecommendations = async (preferences) => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await axios.post(`${API_BASE_URL}/recommendations`, preferences);
      setRecommendations(response.data);
    } catch (err) {
      console.error('Error getting recommendations:', err);
      setError(err.response?.data?.error || 'Failed to get recommendations');
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setRecommendations(null);
    setError(null);
  };

  if (error && !systemReady) {
    return (
      <div className="app">
        <div className="error-container">
          <div className="error-card">
            <h2>🚫 System Error</h2>
            <p>{error}</p>
            <div className="error-instructions">
              <h3>To fix this:</h3>
              <ol>
                <li>Make sure you have the backend running on localhost:5000</li>
                <li>Navigate to the backend directory</li>
                <li>Activate your virtual environment</li>
                <li>Run: <code>python app.py</code></li>
              </ol>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="app">
      <header className="app-header">
        <div className="header-content">
          <h1>🏠 Bangalore Area Recommendation System</h1>
          <p>AI-powered recommendations for finding your perfect neighborhood</p>
          
          {systemReady && datasetInfo && (
            <div className="system-status">
              <div className="status-item">
                <span className="status-icon">✅</span>
                <span>System Ready</span>
              </div>
              <div className="status-item">
                <span className="status-icon">📊</span>
                <span>{datasetInfo.total_areas} Areas Available</span>
              </div>
              <div className="status-item">
                <span className="status-icon">🤖</span>
                <span>ML Model Loaded</span>
              </div>
            </div>
          )}
        </div>
      </header>

      <main className="app-main">
        {!systemReady ? (
          <div className="loading-system">
            <LoadingSpinner />
            <p>Initializing ML system...</p>
          </div>
        ) : !recommendations ? (
          <div className="preferences-section">
            <UserPreferences 
              onSubmit={handleGetRecommendations}
              loading={loading}
              datasetInfo={datasetInfo}
            />
            
            {datasetInfo && (
              <div className="dataset-overview">
                <h2>📊 Dataset Overview</h2>
                <div className="overview-grid">
                  <div className="overview-card">
                    <h3>Price Range</h3>
                    <p>₹{datasetInfo.price_range.min.toLocaleString()} - ₹{datasetInfo.price_range.max.toLocaleString()} per sqft</p>
                    <small>Median: ₹{datasetInfo.price_range.median.toLocaleString()}</small>
                  </div>
                  <div className="overview-card">
                    <h3>Air Quality</h3>
                    <p>AQI: {datasetInfo.aqi_range.min} - {datasetInfo.aqi_range.max}</p>
                    <small>Average: {datasetInfo.aqi_range.mean.toFixed(0)}</small>
                  </div>
                  <div className="overview-card">
                    <h3>Area Types</h3>
                    <p>{datasetInfo.area_types.join(', ')}</p>
                  </div>
                  <div className="overview-card">
                    <h3>Metro Access</h3>
                    <p>{datasetInfo.metro_access_areas} areas with metro</p>
                    <small>Out of {datasetInfo.total_areas} total</small>
                  </div>
                </div>
              </div>
            )}

            {featureImportance && (
              <div className="feature-importance">
                <h2>🔍 Key Factors in Our ML Model</h2>
                <div className="features-grid">
                  {featureImportance.feature_importance.slice(0, 6).map((feature, index) => (
                    <div key={index} className="feature-card">
                      <div className="feature-name">{feature.feature.replace(/_/g, ' ')}</div>
                      <div className="feature-bar">
                        <div 
                          className="feature-fill" 
                          style={{ width: `${(feature.importance * 100)}%` }}
                        ></div>
                      </div>
                      <div className="feature-value">{(feature.importance * 100).toFixed(1)}%</div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        ) : (
          <div className="results-section">
            <Recommendations 
              data={recommendations}
              onReset={handleReset}
            />
          </div>
        )}

        {error && (
          <div className="error-message">
            <h3>❌ Error</h3>
            <p>{error}</p>
          </div>
        )}
      </main>

      <footer className="app-footer">
        <p>Built with React + Flask + Scikit-learn | Bangalore Area Recommendation System</p>
      </footer>
    </div>
  );
}

export default App;