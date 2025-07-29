import React, { useState } from 'react';

const Recommendations = ({ data, onReset }) => {
  const [selectedArea, setSelectedArea] = useState(null);
  const [showComparison, setShowComparison] = useState(false);

  const { recommendations, analytics, user_preferences } = data;

  const formatCurrency = (amount) => {
    return `₹${amount.toLocaleString()}`;
  };

  const getScoreColor = (score) => {
    if (score >= 80) return '#10b981'; // Green
    if (score >= 60) return '#f59e0b'; // Yellow
    if (score >= 40) return '#f97316'; // Orange
    return '#ef4444'; // Red
  };

  const getAQIColor = (aqi) => {
    if (aqi <= 50) return '#10b981'; // Green
    if (aqi <= 100) return '#84cc16'; // Light green
    if (aqi <= 150) return '#f59e0b'; // Yellow
    if (aqi <= 200) return '#f97316'; // Orange
    return '#ef4444'; // Red
  };

  const ScoreBar = ({ score, label, color = null }) => (
    <div className="score-bar-container">
      <div className="score-bar-label">
        <span>{label}</span>
        <span className="score-value">{score.toFixed(1)}</span>
      </div>
      <div className="score-bar">
        <div 
          className="score-bar-fill" 
          style={{ 
            width: `${Math.min(score, 100)}%`,
            backgroundColor: color || getScoreColor(score)
          }}
        ></div>
      </div>
    </div>
  );

  const RecommendationCard = ({ area, rank }) => (
    <div className="recommendation-card">
      <div className="card-header">
        <div className="area-info">
          <div className="rank-badge">#{rank}</div>
          <div className="area-details">
            <h3>{area.Area_Name}</h3>
            <span className="area-type">{area.Type}</span>
          </div>
        </div>
        <div className="price-tag">
          {formatCurrency(area.Price_per_sqft)}/sqft
        </div>
      </div>

      <div className="card-content">
        {/* Key Metrics */}
        <div className="metrics-grid">
          <div className="metric">
            <div className="metric-label">🌱 Air Quality</div>
            <div className="metric-value">
              <span style={{ color: getAQIColor(area.Annual_Avg_AQI) }}>
                AQI {area.Annual_Avg_AQI.toFixed(0)}
              </span>
              <small>{area.AQI_Category}</small>
            </div>
          </div>

          <div className="metric">
            <div className="metric-label">🚇 Metro</div>
            <div className="metric-value">
              {area.Metro_Access ? (
                <span style={{ color: '#10b981' }}>✅ Available</span>
              ) : (
                <span style={{ color: '#ef4444' }}>❌ No Access</span>
              )}
            </div>
          </div>

          <div className="metric">
            <div className="metric-label">💼 IT Hubs</div>
            <div className="metric-value">
              <span>{area.IT_Hub_Distance_km.toFixed(1)} km</span>
            </div>
          </div>

          <div className="metric">
            <div className="metric-label">🏫 Schools</div>
            <div className="metric-value">
              <span>{area.School_Rating}/5</span>
            </div>
          </div>

          <div className="metric">
            <div className="metric-label">🛡️ Safety</div>
            <div className="metric-value">
              <span>{area.Safety_Score}/10</span>
            </div>
          </div>

          <div className="metric">
            <div className="metric-label">🌳 Green Space</div>
            <div className="metric-value">
              <span>{area['Green_Space_%'].toFixed(1)}%</span>
            </div>
          </div>
        </div>

        {/* Score Bars */}
        <div className="scores-section">
          <ScoreBar 
            score={area.Combined_Score} 
            label="🎯 Overall Match" 
            color="#8b5cf6"
          />
          <ScoreBar 
            score={area.Preference_Score} 
            label="❤️ Your Preferences" 
            color="#ec4899"
          />
          <ScoreBar 
            score={area.Predicted_Livability} 
            label="🤖 ML Prediction" 
            color="#06b6d4"
          />
          <ScoreBar 
            score={area.Livability_Score} 
            label="⭐ Actual Livability" 
            color="#10b981"
          />
        </div>

        {/* Explanations */}
        <div className="explanations">
          <h4>✨ Why this area matches you:</h4>
          <ul className="explanation-list">
            {area.explanations.map((explanation, index) => (
              <li key={index}>{explanation}</li>
            ))}
          </ul>
        </div>

        <button 
          className="details-btn"
          onClick={() => setSelectedArea(selectedArea === area ? null : area)}
        >
          {selectedArea === area ? 'Hide Details' : 'Show More Details'}
        </button>

        {selectedArea === area && (
          <div className="detailed-info">
            <div className="details-grid">
              <div className="detail-item">
                <strong>🏥 Hospital Distance:</strong>
                <span>{area.Hospital_Distance_km.toFixed(1)} km</span>
              </div>
              <div className="detail-item">
                <strong>🔗 Connectivity Score:</strong>
                <span>{area.Connectivity_Score}/10</span>
              </div>
              <div className="detail-item">
                <strong>📊 ML vs Actual:</strong>
                <span>
                  Predicted: {area.Predicted_Livability.toFixed(1)}, 
                  Actual: {area.Livability_Score.toFixed(1)}
                </span>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );

  return (
    <div className="recommendations-container">
      {/* Header */}
      <div className="results-header">
        <div className="header-content">
          <h2>🎯 Your Personalized Recommendations</h2>
          <p>Based on your preferences and ML analysis</p>
          
          <div className="action-buttons">
            <button className="secondary-btn" onClick={onReset}>
              🔄 New Search
            </button>
            <button 
              className="secondary-btn"
              onClick={() => setShowComparison(!showComparison)}
            >
              📊 {showComparison ? 'Hide' : 'Show'} Comparison
            </button>
          </div>
        </div>
      </div>

      {/* Analytics Summary */}
      <div className="analytics-summary">
        <div className="summary-cards">
          <div className="summary-card">
            <div className="summary-icon">🔍</div>
            <div className="summary-content">
              <div className="summary-number">{analytics.total_areas_searched}</div>
              <div className="summary-label">Areas Analyzed</div>
            </div>
          </div>
          
          <div className="summary-card">
            <div className="summary-icon">💰</div>
            <div className="summary-content">
              <div className="summary-number">{analytics.areas_within_budget}</div>
              <div className="summary-label">Within Budget</div>
            </div>
          </div>
          
          <div className="summary-card">
            <div className="summary-icon">⭐</div>
            <div className="summary-content">
              <div className="summary-number">{analytics.recommendations_count}</div>
              <div className="summary-label">Top Matches</div>
            </div>
          </div>

          <div className="summary-card">
            <div className="summary-icon">🎯</div>
            <div className="summary-content">
              <div className="summary-number">{formatCurrency(user_preferences.max_budget)}</div>
              <div className="summary-label">Your Budget</div>
            </div>
          </div>
        </div>
      </div>

      {/* Comparison View */}
      {showComparison && recommendations.length > 1 && (
        <div className="comparison-section">
          <h3>📊 Quick Comparison</h3>
          <div className="comparison-table">
            <div className="comparison-header">
              <div className="comparison-cell">Area</div>
              <div className="comparison-cell">Price</div>
              <div className="comparison-cell">AQI</div>
              <div className="comparison-cell">Metro</div>
              <div className="comparison-cell">Schools</div>
              <div className="comparison-cell">Safety</div>
              <div className="comparison-cell">Match Score</div>
            </div>
            {recommendations.slice(0, 5).map((area, index) => (
              <div key={index} className="comparison-row">
                <div className="comparison-cell">
                  <strong>{area.Area_Name}</strong>
                  <small>{area.Type}</small>
                </div>
                <div className="comparison-cell">{formatCurrency(area.Price_per_sqft)}</div>
                <div className="comparison-cell">
                  <span style={{ color: getAQIColor(area.Annual_Avg_AQI) }}>
                    {area.Annual_Avg_AQI.toFixed(0)}
                  </span>
                </div>
                <div className="comparison-cell">
                  {area.Metro_Access ? '✅' : '❌'}
                </div>
                <div className="comparison-cell">{area.School_Rating}/5</div>
                <div className="comparison-cell">{area.Safety_Score}/10</div>
                <div className="comparison-cell">
                  <span style={{ color: getScoreColor(area.Combined_Score) }}>
                    {area.Combined_Score.toFixed(1)}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Recommendations List */}
      <div className="recommendations-list">
        {recommendations.length === 0 ? (
          <div className="no-results">
            <h3>😔 No areas found matching your criteria</h3>
            <p>Try adjusting your budget or preferences to see more options.</p>
            <button className="primary-btn" onClick={onReset}>
              🔄 Adjust Preferences
            </button>
          </div>
        ) : (
          <>
            {recommendations.map((area, index) => (
              <RecommendationCard 
                key={area.Area_Name} 
                area={area} 
                rank={index + 1}
              />
            ))}
          </>
        )}
      </div>

      {/* Footer Tips */}
      <div className="results-footer">
        <div className="tips-section">
          <h3>💡 Tips for choosing your area:</h3>
          <div className="tips-grid">
            <div className="tip">
              <strong>🚇 Consider Future Metro Plans:</strong>
              <span>Even if metro isn't available now, check upcoming metro lines</span>
            </div>
            <div className="tip">
              <strong>🏥 Healthcare Access:</strong>
              <span>Look for areas with good hospital connectivity</span>
            </div>
            <div className="tip">
              <strong>🌱 Air Quality Trends:</strong>
              <span>AQI can vary seasonally, consider year-round patterns</span>
            </div>
            <div className="tip">
              <strong>📈 Growth Potential:</strong>
              <span>Areas under development may offer better long-term value</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Recommendations;