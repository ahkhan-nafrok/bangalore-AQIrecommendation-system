import React, { useState } from 'react';
import LoadingSpinner from './LoadingSpinner';

const UserPreferences = ({ onSubmit, loading, datasetInfo }) => {
  const [preferences, setPreferences] = useState({
    max_budget: '',
    aqi_preference: 'Good',
    metro_access: false,
    family_priority: false,
    green_space_priority: false,
    top_n: 5
  });

  const [errors, setErrors] = useState({});

  const handleInputChange = (e) => {
    const { name, value, type, checked } = e.target;
    setPreferences(prev => ({
      ...prev,
      [name]: type === 'checkbox' ? checked : value
    }));
    
    // Clear error when user starts typing
    if (errors[name]) {
      setErrors(prev => ({ ...prev, [name]: null }));
    }
  };

  const validateForm = () => {
    const newErrors = {};

    if (!preferences.max_budget || parseFloat(preferences.max_budget) <= 0) {
      newErrors.max_budget = 'Please enter a valid budget amount';
    }

    if (datasetInfo && parseFloat(preferences.max_budget) < datasetInfo.price_range.min) {
      newErrors.max_budget = `Budget too low. Minimum available: ₹${datasetInfo.price_range.min.toLocaleString()}`;
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    
    if (!validateForm()) {
      return;
    }

    const formattedPreferences = {
      ...preferences,
      max_budget: parseFloat(preferences.max_budget),
      top_n: parseInt(preferences.top_n)
    };

    onSubmit(formattedPreferences);
  };

  const getBudgetSuggestions = () => {
    if (!datasetInfo) return [];
    
    const { min, max, median, mean } = datasetInfo.price_range;
    return [
      { label: 'Budget Option', value: Math.round(min * 1.1) },
      { label: 'Median Price', value: Math.round(median) },
      { label: 'Average Price', value: Math.round(mean) },
      { label: 'Premium Option', value: Math.round(max * 0.9) }
    ];
  };

  return (
    <div className="preferences-container">
      <div className="preferences-card">
        <h2>🎯 Tell us your preferences</h2>
        <p>Help us find the perfect area for you in Bangalore</p>

        <form onSubmit={handleSubmit} className="preferences-form">
          {/* Budget Section */}
          <div className="form-section">
            <h3>💰 Budget</h3>
            <div className="form-group">
              <label htmlFor="max_budget">Maximum Budget per sq ft (₹)</label>
              <input
                type="number"
                id="max_budget"
                name="max_budget"
                value={preferences.max_budget}
                onChange={handleInputChange}
                placeholder="Enter your maximum budget"
                className={errors.max_budget ? 'error' : ''}
              />
              {errors.max_budget && (
                <div className="error-text">{errors.max_budget}</div>
              )}
              
              {datasetInfo && (
                <div className="budget-suggestions">
                  <p>💡 Quick suggestions based on available data:</p>
                  <div className="suggestion-buttons">
                    {getBudgetSuggestions().map((suggestion, index) => (
                      <button
                        key={index}
                        type="button"
                        className="suggestion-btn"
                        onClick={() => setPreferences(prev => ({ 
                          ...prev, 
                          max_budget: suggestion.value.toString() 
                        }))}
                      >
                        {suggestion.label}: ₹{suggestion.value.toLocaleString()}
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Air Quality Section */}
          <div className="form-section">
            <h3>🌱 Air Quality Preference</h3>
            <div className="form-group">
              <div className="radio-group">
                <label className="radio-option">
                  <input
                    type="radio"
                    name="aqi_preference"
                    value="Good"
                    checked={preferences.aqi_preference === 'Good'}
                    onChange={handleInputChange}
                  />
                  <span className="radio-custom"></span>
                  <div className="radio-content">
                    <strong>Good Air Quality</strong>
                    <small>AQI &lt; 100 (Healthiest option)</small>
                  </div>
                </label>

                <label className="radio-option">
                  <input
                    type="radio"
                    name="aqi_preference"
                    value="Moderate"
                    checked={preferences.aqi_preference === 'Moderate'}
                    onChange={handleInputChange}
                  />
                  <span className="radio-custom"></span>
                  <div className="radio-content">
                    <strong>Moderate Air Quality</strong>
                    <small>AQI &lt; 150 (Acceptable for most)</small>
                  </div>
                </label>

                <label className="radio-option">
                  <input
                    type="radio"
                    name="aqi_preference"
                    value="Don't mind"
                    checked={preferences.aqi_preference === "Don't mind"}
                    onChange={handleInputChange}
                  />
                  <span className="radio-custom"></span>
                  <div className="radio-content">
                    <strong>Don't mind pollution</strong>
                    <small>Any AQI level is fine</small>
                  </div>
                </label>
              </div>
            </div>
          </div>

          {/* Lifestyle Preferences */}
          <div className="form-section">
            <h3>🏙️ Lifestyle Preferences</h3>
            <div className="checkbox-group">
              <label className="checkbox-option">
                <input
                  type="checkbox"
                  name="metro_access"
                  checked={preferences.metro_access}
                  onChange={handleInputChange}
                />
                <span className="checkbox-custom"></span>
                <div className="checkbox-content">
                  <strong>🚇 Metro Access Required</strong>
                  <small>I need easy access to metro stations</small>
                </div>
              </label>

              <label className="checkbox-option">
                <input
                  type="checkbox"
                  name="family_priority"
                  checked={preferences.family_priority}
                  onChange={handleInputChange}
                />
                <span className="checkbox-custom"></span>
                <div className="checkbox-content">
                  <strong>👨‍👩‍👧‍👦 Family-Friendly</strong>
                  <small>Good schools and safe neighborhoods are important</small>
                </div>
              </label>

              <label className="checkbox-option">
                <input
                  type="checkbox"
                  name="green_space_priority"
                  checked={preferences.green_space_priority}
                  onChange={handleInputChange}
                />
                <span className="checkbox-custom"></span>
                <div className="checkbox-content">
                  <strong>🌳 Green Spaces</strong>
                  <small>Parks and green areas are important to me</small>
                </div>
              </label>
            </div>
          </div>

          {/* Number of recommendations */}
          <div className="form-section">
            <h3>📊 Results</h3>
            <div className="form-group">
              <label htmlFor="top_n">Number of recommendations</label>
              <select
                id="top_n"
                name="top_n"
                value={preferences.top_n}
                onChange={handleInputChange}
              >
                <option value="3">3 recommendations</option>
                <option value="5">5 recommendations</option>
                <option value="7">7 recommendations</option>
                <option value="10">10 recommendations</option>
              </select>
            </div>
          </div>

          <button 
            type="submit" 
            className="submit-btn"
            disabled={loading}
          >
            {loading ? (
              <>
                <LoadingSpinner size="small" />
                Analyzing Areas...
              </>
            ) : (
              '🔍 Find My Perfect Area'
            )}
          </button>
        </form>
      </div>
    </div>
  );
};

export default UserPreferences;