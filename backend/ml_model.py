import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import warnings
import joblib
import os
warnings.filterwarnings('ignore')

class BangaloreRecommendationSystem:
    def __init__(self):
        self.df = None
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = []
        self.user_profile_scaler = StandardScaler()
        self.model_path = os.path.join('models', 'bangalore_model.pkl')
        self.scaler_path = os.path.join('models', 'scaler.pkl')
        self.encoders_path = os.path.join('models', 'label_encoders.pkl')
        self.features_path = os.path.join('models', 'feature_columns.pkl')  # NEW: Save feature columns

    def load_data(self, file_path):
        """Load the actual CSV dataset"""
        try:
            print(f"📊 Loading dataset from: {file_path}")
            self.df = pd.read_csv(file_path)
            print(f"✅ Dataset loaded successfully!")
            print(f"📋 Shape: {self.df.shape}")
            print(f"📋 Columns: {list(self.df.columns)}")

            # Display basic info
            print("\n🔍 Dataset Overview:")
            print(self.df.head(3))
            print(f"\n📈 Dataset Info:")
            print(f"   • Total areas: {len(self.df)}")
            print(f"   • Features: {len(self.df.columns)}")

            # Check for missing values
            missing_vals = self.df.isnull().sum()
            if missing_vals.any():
                print(f"\n⚠️ Missing values found:")
                print(missing_vals[missing_vals > 0])
                # Fill missing values
                for col in missing_vals[missing_vals > 0].index:
                    if self.df[col].dtype in ['int64', 'float64']:
                        self.df[col].fillna(self.df[col].median(), inplace=True)
                    else:
                        self.df[col].fillna(self.df[col].mode()[0], inplace=True)
                print("✅ Missing values filled!")
            else:
                print("✅ No missing values found!")

            return True

        except FileNotFoundError:
            print(f"❌ Error: File '{file_path}' not found!")
            print("💡 Please ensure the CSV file is in the correct location")
            return False
        except Exception as e:
            print(f"❌ Error loading dataset: {str(e)}")
            return False

    def prepare_features(self):
        """Prepare features for ML model with interaction features"""
        print("🔧 Preparing features for ML model...")

        # Display column info
        print(f"📋 All columns: {list(self.df.columns)}")
        print(f"📋 Data types:\n{self.df.dtypes}")

        # Handle categorical variables
        categorical_columns = self.df.select_dtypes(include=['object']).columns
        print(f"📋 Categorical columns found: {list(categorical_columns)}")

        # Encode categorical variables
        for col in categorical_columns:
            if col not in ['Area', 'Area_Name']:  # Don't encode area names
                le = LabelEncoder()
                self.df[f'{col}_encoded'] = le.fit_transform(self.df[col].astype(str))
                self.label_encoders[col] = le
                print(f"   ✅ Encoded {col}")

        # Handle boolean columns
        bool_columns = self.df.select_dtypes(include=['bool']).columns
        for col in bool_columns:
            self.df[f'{col}_int'] = self.df[col].astype(int)
            print(f"   ✅ Converted {col} to integer")

        # Add interaction features
        if 'School_Rating' in self.df.columns and 'Safety_Score' in self.df.columns:
            self.df['School_Safety'] = self.df['School_Rating'] * self.df['Safety_Score']
        if 'Green_Space_%' in self.df.columns and 'Connectivity_Score' in self.df.columns:
            self.df['Green_Connectivity'] = self.df['Green_Space_%'] * self.df['Connectivity_Score']
        print("   ✅ Added interaction features")

        # Define feature columns - only use numerical and encoded columns
        exclude_cols = ['Area', 'Area_Name', 'Livability_Score']

        # Get numerical columns
        numerical_cols = self.df.select_dtypes(include=['int64', 'float64']).columns
        numerical_cols = [col for col in numerical_cols if col not in exclude_cols]

        # Get encoded columns
        encoded_cols = [col for col in self.df.columns if col.endswith('_encoded') or col.endswith('_int')]

        # Combine numerical and encoded columns
        self.feature_columns = list(numerical_cols) + encoded_cols
        
        # Add interaction features if they exist
        if 'School_Safety' in self.df.columns:
            self.feature_columns.append('School_Safety')
        if 'Green_Connectivity' in self.df.columns:
            self.feature_columns.append('Green_Connectivity')

        # Remove duplicates and ensure no string columns
        self.feature_columns = list(set(self.feature_columns))

        # Final check - make sure all feature columns are numeric
        final_features = []
        for col in self.feature_columns:
            if col in self.df.columns and self.df[col].dtype in ['int64', 'float64']:
                final_features.append(col)
            else:
                print(f"⚠️ Removing non-numeric column: {col}")

        self.feature_columns = final_features
        print(f"📋 Final feature columns: {self.feature_columns}")
        print(f"📊 Total features: {len(self.feature_columns)}")

    def train_model(self):
        """Train ML model to predict livability scores with regularization"""
        print("🤖 Training ML model...")

        # Check if trained model exists
        if (os.path.exists(self.model_path) and 
            os.path.exists(self.scaler_path) and 
            os.path.exists(self.encoders_path) and
            os.path.exists(self.features_path)):  # NEW: Check for feature columns file
            
            print("🔄 Loading pre-trained model...")
            try:
                self.model = joblib.load(self.model_path)
                self.scaler = joblib.load(self.scaler_path)
                self.label_encoders = joblib.load(self.encoders_path)
                self.feature_columns = joblib.load(self.features_path)  # NEW: Load feature columns
                print("✅ Pre-trained model loaded successfully!")
                print(f"📋 Loaded feature columns: {self.feature_columns}")
                return None
            except Exception as e:
                print(f"⚠️ Could not load pre-trained model: {e}")
                print("Training new model...")

        # Prepare features and target
        X = self.df[self.feature_columns]
        y = self.df['Livability_Score']

        print(f"📊 Training data shape: {X.shape}")
        print(f"📊 Target shape: {y.shape}")
        print(f"📋 Feature columns during training: {self.feature_columns}")

        # Handle missing values
        if X.isnull().sum().sum() > 0:
            print("⚠️ Found missing values, filling with median...")
            X = X.fillna(X.median())

        # Split data
        if len(X) > 4:  # Need at least 5 samples for splitting
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
        else:
            # Use all data for training if dataset is very small
            X_train, X_test, y_train, y_test = X, X, y, y
            print("⚠️ Small dataset - using all data for training and testing")

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train Gradient Boosting model with regularization
        self.model = GradientBoostingRegressor(
            n_estimators=500,
            learning_rate=0.1,
            max_depth=3,
            min_samples_leaf=5,
            validation_fraction=0.1,
            n_iter_no_change=10,
            tol=0.01,
            random_state=42
        )

        self.model.fit(X_train_scaled, y_train)

        # Evaluate model
        y_pred_train = self.model.predict(X_train_scaled)
        y_pred_test = self.model.predict(X_test_scaled)

        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        test_mse = mean_squared_error(y_test, y_pred_test)

        print(f"✅ Model trained successfully!")
        print(f"📊 Training R² Score: {train_r2:.3f}")
        print(f"📊 Testing R² Score: {test_r2:.3f}")
        print(f"📊 Test MSE: {test_mse:.3f}")

        # Save the trained model
        os.makedirs('models', exist_ok=True)
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.scaler, self.scaler_path)
        joblib.dump(self.label_encoders, self.encoders_path)
        joblib.dump(self.feature_columns, self.features_path)  # NEW: Save feature columns
        print(f"💾 Model saved to {self.model_path}")
        print(f"💾 Feature columns saved to {self.features_path}")

        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        print(f"\n🔍 Top 5 Most Important Features:")
        for i, (_, row) in enumerate(feature_importance.head().iterrows()):
            print(f"   {i+1}. {row['feature']}: {row['importance']:.3f}")

        return feature_importance

    def calculate_preference_score(self, area_row, user_preferences):
        """Calculate user preference score for a given area"""
        score = 0
        weights = {
            'budget': 0.25,
            'aqi': 0.2,
            'metro': 0.15,
            'family': 0.2,
            'green': 0.2
        }

        price = area_row['Price_per_sqft']
        max_budget = user_preferences['max_budget']

        # STRICT BUDGET ENFORCEMENT - eliminate areas over budget
        if price > max_budget:
            return 0

        # Budget score
        budget_score = (max_budget - price) / max_budget
        score += weights['budget'] * budget_score

        # AQI Score logic
        aqi_val = area_row['Annual_Avg_AQI']
        if user_preferences['aqi_preference'] == 'Good':
            aqi_score = max(0, (100 - aqi_val) / 100)
        elif user_preferences['aqi_preference'] == 'Moderate':
            aqi_score = max(0, (150 - aqi_val) / 150)
        else:
            aqi_score = max(0.3, (200 - aqi_val) / 200)
        score += weights['aqi'] * aqi_score

        # Metro Access Score
        if user_preferences['metro_access']:
            metro_score = 1 if area_row['Metro_Access'] else 0.2
        else:
            metro_score = 1 if area_row['Metro_Access'] else 0.7
        score += weights['metro'] * metro_score

        # Family Score
        if user_preferences['family_priority']:
            family_score = (area_row['School_Rating'] / 5 + area_row['Safety_Score'] / 10) / 2
        else:
            family_score = ((area_row['School_Rating'] / 5) * 0.3 + (area_row['Safety_Score'] / 10) * 0.7)
        score += weights['family'] * family_score

        # Green Space Score
        if user_preferences['green_space_priority']:
            green_score = area_row['Green_Space_%'] / 100
        else:
            green_score = (area_row['Green_Space_%'] / 100) * 0.4 + 0.3
        score += weights['green'] * green_score

        return score * 100  # Scale to 0–100


    def get_ml_recommendations(self, user_preferences, top_n=5):
        """Get recommendations using ML model + preference scoring"""
        print("🔍 Generating ML-based recommendations...")

        recommendations = []

        # FIXED: Ensure feature columns are available
        if not self.feature_columns:
            print("❌ Feature columns not loaded. Running prepare_features()...")
            self.prepare_features()

        # FIXED: Debug feature column information
        print(f"📋 Using feature columns: {self.feature_columns}")
        print(f"📋 Available DataFrame columns: {list(self.df.columns)}")

        # FIXED: Check if all feature columns exist in dataframe
        missing_features = [col for col in self.feature_columns if col not in self.df.columns]
        if missing_features:
            print(f"❌ Missing features in DataFrame: {missing_features}")
            # Remove missing features from feature_columns
            self.feature_columns = [col for col in self.feature_columns if col in self.df.columns]
            print(f"📋 Updated feature columns: {self.feature_columns}")

        # Get ML predictions for all areas
        try:
            X_all = self.df[self.feature_columns]
            print(f"📊 Prediction data shape: {X_all.shape}")
            print(f"📊 Feature columns count: {len(self.feature_columns)}")

            # Handle missing values in features
            if X_all.isnull().sum().sum() > 0:
                print("⚠️ Found missing values in prediction data, filling with median...")
                X_all = X_all.fillna(X_all.median())

            # FIXED: Add debugging information before scaling
            print(f"📊 Data types in X_all:")
            print(X_all.dtypes)
            
            # Ensure all columns are numeric
            for col in X_all.columns:
                if X_all[col].dtype == 'object':
                    print(f"⚠️ Converting non-numeric column {col} to numeric")
                    X_all[col] = pd.to_numeric(X_all[col], errors='coerce')
                    X_all[col] = X_all[col].fillna(X_all[col].median())

            X_all_scaled = self.scaler.transform(X_all)
            predicted_livability = self.model.predict(X_all_scaled)

        except ValueError as e:
            print(f"❌ Error during prediction: {str(e)}")
            print("🔧 Attempting to fix feature mismatch...")
            
            # Get scaler's expected features
            if hasattr(self.scaler, 'feature_names_in_'):
                expected_features = self.scaler.feature_names_in_
                print(f"📋 Scaler expects features: {list(expected_features)}")
                
                # Reorder and filter features to match scaler expectations
                available_features = [f for f in expected_features if f in self.df.columns]
                missing_from_df = [f for f in expected_features if f not in self.df.columns]
                
                print(f"📋 Available features: {available_features}")
                print(f"❌ Missing features: {missing_from_df}")
                
                if missing_from_df:
                    print("❌ Cannot proceed - required features missing from dataset")
                    return []
                
                # Use only available features in correct order
                X_all = self.df[available_features]
                if X_all.isnull().sum().sum() > 0:
                    X_all = X_all.fillna(X_all.median())
                
                X_all_scaled = self.scaler.transform(X_all)
                predicted_livability = self.model.predict(X_all_scaled)
            else:
                print("❌ Cannot determine expected features. Please retrain the model.")
                return []

        # Calculate preference scores and combine with ML predictions
        for idx, area in self.df.iterrows():
            preference_score = self.calculate_preference_score(area, user_preferences)

            # Skip areas that are over budget
            if preference_score == 0:
                continue

            # Combine ML prediction with preference score
            ml_score = predicted_livability[idx]
            combined_score = 0.6 * preference_score + 0.4 * ml_score

            # Get area information
            area_name = area.get('Area_Name', area.get('Area', f'Area_{idx}'))
            area_type = area.get('Type', 'Unknown')
            aqi_category = area.get('AQI_Category', 'Unknown')

            recommendations.append({
                'Area_Name': area_name,
                'Type': area_type,
                'Price_per_sqft': float(area['Price_per_sqft']),
                'Annual_Avg_AQI': float(area['Annual_Avg_AQI']),
                'AQI_Category': aqi_category,
                'Green_Space_%': float(area.get('Green_Space_%', 0)),
                'Metro_Access': bool(area.get('Metro_Access', False)),
                'IT_Hub_Distance_km': float(area.get('IT_Hub_Distance_km', 0)),
                'Hospital_Distance_km': float(area.get('Hospital_Distance_km', 0)),
                'School_Rating': float(area.get('School_Rating', 0)),
                'Safety_Score': float(area.get('Safety_Score', 0)),
                'Connectivity_Score': float(area.get('Connectivity_Score', 0)),
                'Livability_Score': float(area['Livability_Score']),
                'Predicted_Livability': float(ml_score),
                'Preference_Score': float(preference_score),
                'Combined_Score': float(combined_score)
            })

        # Sort by combined score
        recommendations.sort(key=lambda x: x['Combined_Score'], reverse=True)

        # Return top N recommendations
        final_recommendations = recommendations[:top_n]

        print(f"✅ Found {len(final_recommendations)} suitable areas")
        return final_recommendations

    def explain_recommendation(self, area, user_preferences):
        """Generate detailed explanation for recommendations"""
        explanations = []

        price = area['Price_per_sqft']
        max_budget = user_preferences['max_budget']
        budget_ratio = price / max_budget

        # Budget explanation
        if budget_ratio <= 0.7:
            explanations.append(f"💰 Excellent value at ₹{price:,.0f}/sqft ({budget_ratio:.0%} of budget)")
        elif budget_ratio <= 0.9:
            explanations.append(f"💰 Good value at ₹{price:,.0f}/sqft ({budget_ratio:.0%} of budget)")
        else:
            explanations.append(f"💰 Within budget at ₹{price:,.0f}/sqft ({budget_ratio:.0%} of budget)")

        # AQI explanation
        aqi_val = area['Annual_Avg_AQI']
        if aqi_val <= 50:
            explanations.append(f"🌱 Excellent air quality (AQI: {aqi_val:.0f})")
        elif aqi_val <= 100:
            explanations.append(f"🌱 Good air quality (AQI: {aqi_val:.0f})")
        elif aqi_val <= 150:
            explanations.append(f"🌱 Moderate air quality (AQI: {aqi_val:.0f})")
        else:
            explanations.append(f"🌱 Poor air quality (AQI: {aqi_val:.0f})")

        # Metro access
        if area['Metro_Access']:
            explanations.append("🚇 Metro connectivity available")
        else:
            explanations.append("🚇 No direct metro access")

        # IT hub proximity
        it_dist = area['IT_Hub_Distance_km']
        if it_dist <= 5:
            explanations.append(f"💼 Very close to IT hubs ({it_dist:.1f} km)")
        elif it_dist <= 10:
            explanations.append(f"💼 Reasonable commute to IT hubs ({it_dist:.1f} km)")
        else:
            explanations.append(f"💼 Far from IT hubs ({it_dist:.1f} km)")

        # School rating
        school_rating = area['School_Rating']
        if school_rating >= 4:
            explanations.append(f"🏫 Excellent schools (Rating: {school_rating}/5)")
        elif school_rating >= 3:
            explanations.append(f"🏫 Good schools (Rating: {school_rating}/5)")
        else:
            explanations.append(f"🏫 Average schools (Rating: {school_rating}/5)")

        # Safety score
        safety = area['Safety_Score']
        if safety >= 8:
            explanations.append(f"🛡️ Very safe area (Safety: {safety}/10)")
        elif safety >= 6:
            explanations.append(f"🛡️ Safe area (Safety: {safety}/10)")
        else:
            explanations.append(f"🛡️ Moderate safety (Safety: {safety}/10)")

        # Green space
        green_pct = area['Green_Space_%']
        if green_pct >= 25:
            explanations.append(f"🌳 Excellent green coverage ({green_pct:.1f}%)")
        elif green_pct >= 15:
            explanations.append(f"🌳 Good green coverage ({green_pct:.1f}%)")
        else:
            explanations.append(f"🌳 Limited green space ({green_pct:.1f}%)")

        # ML prediction
        explanations.append(f"🤖 ML Predicted Livability: {area['Predicted_Livability']:.1f}/100")

        return explanations