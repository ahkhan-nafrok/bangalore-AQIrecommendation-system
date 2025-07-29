from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys
from ml_model import BangaloreRecommendationSystem
import traceback

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Global variable to store the ML system
ml_system = None

def initialize_ml_system():
    """Initialize the ML system with data"""
    global ml_system
    try:
        print("🚀 Initializing ML Recommendation System...")
        ml_system = BangaloreRecommendationSystem()
        
        # Load data
        data_path = os.path.join('data', 'Bangalore_data.csv')
        if not os.path.exists(data_path):
            print(f"❌ Error: Data file not found at {data_path}")
            return False
            
        if not ml_system.load_data(data_path):
            print("❌ Failed to load data")
            return False
        
        # Prepare features and train model
        ml_system.prepare_features()
        ml_system.train_model()
        
        print("✅ ML System initialized successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error initializing ML system: {str(e)}")
        traceback.print_exc()
        return False

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'ml_system_ready': ml_system is not None
    })

@app.route('/api/dataset-info', methods=['GET'])
def get_dataset_info():
    """Get information about the loaded dataset"""
    if ml_system is None or ml_system.df is None:
        return jsonify({'error': 'ML system not initialized'}), 500
    
    try:
        df = ml_system.df
        
        # Get basic stats
        price_stats = df['Price_per_sqft'].describe()
        aqi_stats = df['Annual_Avg_AQI'].describe()
        
        # Get unique values for categorical columns
        unique_types = df['Type'].unique().tolist() if 'Type' in df.columns else []
        unique_aqi_categories = df['AQI_Category'].unique().tolist() if 'AQI_Category' in df.columns else []
        
        # Metro access distribution
        metro_access_count = df['Metro_Access'].sum() if 'Metro_Access' in df.columns else 0
        
        info = {
            'total_areas': len(df),
            'columns': df.columns.tolist(),
            'price_range': {
                'min': float(price_stats['min']),
                'max': float(price_stats['max']),
                'mean': float(price_stats['mean']),
                'median': float(price_stats['50%'])
            },
            'aqi_range': {
                'min': float(aqi_stats['min']),
                'max': float(aqi_stats['max']),
                'mean': float(aqi_stats['mean'])
            },
            'area_types': unique_types,
            'aqi_categories': unique_aqi_categories,
            'metro_access_areas': int(metro_access_count),
            'sample_areas': df[['Area_Name', 'Type', 'Price_per_sqft', 'Annual_Avg_AQI']].head(5).to_dict('records')
        }
        
        return jsonify(info)
        
    except Exception as e:
        return jsonify({'error': f'Error getting dataset info: {str(e)}'}), 500

@app.route('/api/recommendations', methods=['POST'])
def get_recommendations():
    """Get ML-based recommendations based on user preferences"""
    if ml_system is None:
        return jsonify({'error': 'ML system not initialized'}), 500
    
    try:
        # Get user preferences from request
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['max_budget', 'aqi_preference', 'metro_access', 'family_priority', 'green_space_priority']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        user_preferences = {
            'max_budget': float(data['max_budget']),
            'aqi_preference': data['aqi_preference'],
            'metro_access': bool(data['metro_access']),
            'family_priority': bool(data['family_priority']),
            'green_space_priority': bool(data['green_space_priority'])
        }
        
        # Get recommendations
        top_n = data.get('top_n', 5)
        recommendations = ml_system.get_ml_recommendations(user_preferences, top_n=top_n)
        
        # Add explanations for each recommendation
        for rec in recommendations:
            rec['explanations'] = ml_system.explain_recommendation(rec, user_preferences)
        
        # Get some analytics
        total_areas = len(ml_system.df)
        within_budget = len([r for r in recommendations if r['Price_per_sqft'] <= user_preferences['max_budget']])
        
        response = {
            'recommendations': recommendations,
            'analytics': {
                'total_areas_searched': total_areas,
                'areas_within_budget': within_budget,
                'recommendations_count': len(recommendations)
            },
            'user_preferences': user_preferences
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Error in recommendations: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': f'Error generating recommendations: {str(e)}'}), 500

@app.route('/api/area-details/<area_name>', methods=['GET'])
def get_area_details(area_name):
    """Get detailed information about a specific area"""
    if ml_system is None:
        return jsonify({'error': 'ML system not initialized'}), 500
    
    try:
        # Find the area in the dataset
        area_data = ml_system.df[ml_system.df['Area_Name'] == area_name]
        
        if area_data.empty:
            return jsonify({'error': f'Area "{area_name}" not found'}), 404
        
        area = area_data.iloc[0].to_dict()
        
        # Convert numpy types to Python types for JSON serialization
        for key, value in area.items():
            if hasattr(value, 'item'):  # numpy types have .item() method
                area[key] = value.item()
        
        return jsonify(area)
        
    except Exception as e:
        return jsonify({'error': f'Error getting area details: {str(e)}'}), 500

@app.route('/api/feature-importance', methods=['GET'])
def get_feature_importance():
    """Get feature importance from the trained model"""
    if ml_system is None or ml_system.model is None:
        return jsonify({'error': 'ML model not trained'}), 500
    
    try:
        feature_importance = []
        for i, feature in enumerate(ml_system.feature_columns):
            importance = float(ml_system.model.feature_importances_[i])
            feature_importance.append({
                'feature': feature,
                'importance': importance
            })
        
        # Sort by importance
        feature_importance.sort(key=lambda x: x['importance'], reverse=True)
        
        return jsonify({
            'feature_importance': feature_importance[:10],  # Top 10 features
            'total_features': len(feature_importance)
        })
        
    except Exception as e:
        return jsonify({'error': f'Error getting feature importance: {str(e)}'}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("🌟 Starting Bangalore Recommendation System Backend...")
    
    # Initialize ML system
    if initialize_ml_system():
        print("🚀 Backend ready! Starting Flask server...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("❌ Failed to initialize ML system. Please check your data file.")
        sys.exit(1)