import numpy as np
import pandas as pd
import streamlit as st
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import mysql.connector
from mysql.connector import Error
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'Si#200805',
    'database': 'fitness_tracker_db'
}

# Database connection function
def create_db_connection():
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        return connection
    except Error as e:
        st.error(f"Error connecting to database: {e}")
        return None

# Function to create database table
def create_table():
    try:
        connection = create_db_connection()
        if connection:
            cursor = connection.cursor()
            
            create_table_query = """
            CREATE TABLE IF NOT EXISTS gym_records (
                record_id INT AUTO_INCREMENT PRIMARY KEY,
                age INT,
                gender VARCHAR(10),
                weight_kg FLOAT,
                height_m FLOAT,
                max_bpm INT,
                avg_bpm INT,
                resting_bpm INT,
                session_duration_hours FLOAT,
                calories_burned FLOAT,
                workout_type VARCHAR(50),
                fat_percentage FLOAT,
                water_intake_liters FLOAT,
                workout_frequency_days_week INT,
                experience_level INT,
                bmi FLOAT
            )
            """
            cursor.execute(create_table_query)
            connection.commit()
            return True
    except Error as e:
        st.error(f"Error creating table: {e}")
        return False
    finally:
        if connection:
            connection.close()

# Function to import CSV data to database
def import_csv_to_db():
    try:
        # Read the CSV file
        df = pd.read_csv('gym_members_exercise_tracking_synthetic_data.csv')
        
        # Print the columns to debug
        print("CSV columns:", df.columns.tolist())
        
        # Handle any NaN values before database insertion
        df = df.fillna(0)  # Replace NaN with 0 or appropriate default values
        
        # Convert DataFrame to list of tuples
        records = []
        for _, row in df.iterrows():
            record = (
                int(row['Age']),
                str(row['Gender']),
                float(row['Weight (kg)']),
                float(row['Height (m)']),
                int(row['Max_BPM']),
                int(row['Avg_BPM']),
                int(row['Resting_BPM']),
                float(row['Session_Duration (hours)']),
                float(row['Calories_Burned']),
                str(row['Workout_Type']),
                float(row['Fat_Percentage']),
                float(row['Water_Intake (liters)']),
                int(row['Workout_Frequency (days/week)']),
                int(row['Experience_Level']),
                float(row['BMI'])
            )
            records.append(record)
        
        connection = create_db_connection()
        if connection:
            cursor = connection.cursor()
            
            # Clear existing data
            cursor.execute("TRUNCATE TABLE gym_records")
            
            # Prepare insert query
            insert_query = """
            INSERT INTO gym_records (
                Age, Gender, Weight_kg, Height_m, Max_BPM, Avg_BPM,
                Resting_BPM, Session_Duration_hours, Calories_Burned,
                Workout_Type, Fat_Percentage, Water_Intake_liters,
                Workout_Frequency_days_week, Experience_Level, BMI
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            # Bulk insert
            cursor.executemany(insert_query, records)
            connection.commit()
            print(f"Successfully inserted {len(records)} records")
            st.success(f"Data imported successfully! {len(records)} records inserted.")
            return True
    except Exception as e:
        print(f"Error in import_csv_to_db: {str(e)}")
        st.error(f"Error importing data: {str(e)}")
        return False
    finally:
        if connection:
            connection.close()

# Function to load data from database
def load_data():
    try:
        connection = create_db_connection()
        if connection:
            query = "SELECT * FROM gym_records"
            df = pd.read_sql(query, connection)
            
            # Rename columns to match the expected format
            column_mapping = {
                'age': 'Age',
                'gender': 'Gender',
                'weight_kg': 'Weight (kg)',
                'height_m': 'Height (m)',
                'max_bpm': 'Max_BPM',
                'avg_bpm': 'Avg_BPM',
                'resting_bpm': 'Resting_BPM',
                'session_duration_hours': 'Session_Duration (hours)',
                'water_intake_liters': 'Water_Intake (liters)',
                'workout_frequency_days_week': 'Workout_Frequency (days/week)',
                'calories_burned': 'Calories_Burned'
            }
            df = df.rename(columns=column_mapping)
            
            # Print columns to debug
            print("Database loaded columns:", df.columns.tolist())
            
            return df
    except Exception as e:
        print(f"Error in load_data: {str(e)}")
        st.error(f"Error loading data: {str(e)}")
        return None
    finally:
        if connection:
            connection.close()

def verify_database_setup():
    try:
        connection = create_db_connection()
        if connection:
            cursor = connection.cursor()
            
            # Check if table exists
            cursor.execute("""
                SELECT COUNT(*)
                FROM information_schema.tables
                WHERE table_schema = 'fitness_tracker_db'
                AND table_name = 'gym_records'
            """)
            if cursor.fetchone()[0] == 0:
                st.error("Table 'gym_records' does not exist")
                return False
            
            # Check table structure
            cursor.execute("DESCRIBE gym_records")
            columns = cursor.fetchall()
            print("Database table structure:")
            for col in columns:
                print(f"Column: {col[0]}, Type: {col[1]}")
            
            return True
    except Error as e:
        st.error(f"Database verification failed: {e}")
        return False
    finally:
        if connection:
            connection.close()

# Load data
fitness_data = load_data()
if fitness_data is None:
    st.error("Failed to load data")
    st.stop()

# Load the trained model
try:
    model_data = joblib.load('calories_model_with_evaluation.pkl')
    calories_pipeline = model_data['pipeline']
    metrics = model_data['metrics']
    feature_importance = model_data['feature_importance']
    X_test_saved = model_data['X_test']
    y_test_saved = model_data['y_test']
    y_pred_saved = model_data['y_pred']
except:
    try:
        calories_pipeline = joblib.load('calories_model.pkl')
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        st.stop()

# Set page configuration
st.set_page_config(
    page_title="Advanced Fitness Tracker",
    page_icon="🏃‍♂️",
    layout="wide"
)

# Enhanced CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5em;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 1em;
        background: linear-gradient(120deg, #3498db, #2ecc71);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .stat-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    
    .stat-card:hover {
        transform: translateY(-5px);
    }
    
    .metric-container {
        background: rgba(255,255,255,0.9);
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 5px solid #3498db;
    }
    
    .trend-indicator-up {
        color: #2ecc71;
        font-size: 1.2em;
    }
    
    .trend-indicator-down {
        color: #e74c3c;
        font-size: 1.2em;
    }
    
    .insight-box {
        background: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 4px solid #f1c40f;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown('<h1 class="main-header">🏃‍♂️ Advanced Fitness Analytics Dashboard</h1>', unsafe_allow_html=True)

# Create tabs for better organization
tabs = st.tabs(["Dashboard", "Detailed Analysis", "Progress Tracking", "Recommendations", "Health Insights"])

with tabs[0]:  # Dashboard
    # User input collection with improved layout
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 👤 Personal Details")
        age = st.number_input('Age', 18, 80, 30, key='dashboard_age')
        gender = st.selectbox('Gender', ['Male', 'Female'], key='dashboard_gender')
        weight = st.number_input('Weight (kg)', 40.0, 150.0, 70.0, key='dashboard_weight')
        height = st.number_input('Height (m)', 1.4, 2.2, 1.7, key='dashboard_height')

    with col2:
        st.markdown("### ❤️ Heart Rate Metrics")
        max_bpm = st.number_input('Max Heart Rate (BPM)', 120, 200, 180, key='dashboard_max_bpm')
        avg_bpm = st.number_input('Average Heart Rate (BPM)', 60, 180, 140, key='dashboard_avg_bpm')
        resting_bpm = st.number_input('Resting Heart Rate (BPM)', 40, 100, 60, key='dashboard_resting_bpm')

    with col3:
        st.markdown("### 🏋️‍♂️ Workout Details")
        duration = st.number_input('Session Duration (hours)', 0.25, 4.0, 1.0, key='dashboard_duration')
        frequency = st.number_input('Workout Frequency (days/week)', 1, 7, 3, key='dashboard_frequency')
        workout_type = st.selectbox('Workout Type', 
            ['Cardio', 'Strength Training', 'HIIT', 'Yoga', 'Mixed'], 
            key='dashboard_workout_type')

    # Create input DataFrame
    input_df = pd.DataFrame({
        'Age': [age],
        'Gender': [gender],
        'Weight (kg)': [weight],
        'Height (m)': [height],
        'Max_BPM': [max_bpm],
        'Avg_BPM': [avg_bpm],
        'Resting_BPM': [resting_bpm],
        'Session_Duration (hours)': [duration],
        'Workout_Frequency (days/week)': [frequency]
    })

    # Calculate metrics
    bmi = weight / (height ** 2)
    predicted_calories = calories_pipeline.predict(input_df)[0]
    calories_percentile = stats.percentileofscore(fitness_data['Calories_Burned'], predicted_calories)
    
    # Calculate BMI and percentiles with error handling
    try:
        # Calculate BMI
        bmi = weight / (height ** 2)
        
        # Calculate BMI percentile
        fitness_data['BMI'] = fitness_data['Weight (kg)'] / (fitness_data['Height (m)'] ** 2)
        bmi_percentile = float(stats.percentileofscore(fitness_data['BMI'].dropna(), bmi))
        
        # Calculate resting BPM percentile
        resting_bpm_percentile = float(stats.percentileofscore(
            fitness_data['Resting_BPM'].dropna(), 
            resting_bpm
        ))
        
        # Calculate calories percentile
        calories_percentile = float(stats.percentileofscore(
            fitness_data['Calories_Burned'].dropna(), 
            predicted_calories
        ))
        
        # Validate percentiles before calculation
        if (np.isnan(calories_percentile) or 
            np.isnan(bmi_percentile) or 
            np.isnan(resting_bpm_percentile)):
            fitness_score = 50  # Default score if calculations fail
        else:
            # Calculate fitness score
            fitness_score = min(100, int((calories_percentile + 
                                        (100 - bmi_percentile) + 
                                        (100 - resting_bpm_percentile)) / 3))
    except Exception as e:
        st.error(f"Error calculating fitness metrics: {str(e)}")
        fitness_score = 50  # Default score if calculations fail
        calories_percentile = 50
        bmi_percentile = 50
        resting_bpm_percentile = 50

    # Now these variables are guaranteed to have valid values
    st.write(f"Fitness Score: {fitness_score}")

    # Enhanced metrics display
    st.markdown("### 📊 Key Performance Metrics")
    metric_cols = st.columns(4)
    
    with metric_cols[0]:
        st.markdown("""
            <div class="metric-container">
                <h3>Calories Burned</h3>
                <div class="animated-number">🔥 {:.0f}</div>
                <p>Top {:.1f}% of users</p>
            </div>
        """.format(predicted_calories, 100 - calories_percentile), unsafe_allow_html=True)

    with metric_cols[1]:
        bmi_category = (
            'Underweight' if bmi < 18.5 else
            'Normal weight' if bmi < 24.9 else
            'Overweight' if bmi < 29.9 else
            'Obese'
        )
        st.markdown(f"""
            <div class="metric-container">
                <h3>BMI Status</h3>
                <div class="animated-number">⚖️ {bmi:.1f}</div>
                <p>{bmi_category}</p>
            </div>
        """, unsafe_allow_html=True)

    with metric_cols[2]:
        heart_health = (
            'Excellent' if resting_bpm < 60 else
            'Good' if resting_bpm < 70 else
            'Average' if resting_bpm < 80 else
            'Need Improvement'
        )
        st.markdown(f"""
            <div class="metric-container">
                <h3>Heart Health</h3>
                <div class="animated-number">❤️ {heart_health}</div>
                <p>Based on resting HR</p>
            </div>
        """, unsafe_allow_html=True)

    with metric_cols[3]:
        st.markdown(f"""
            <div class="metric-container">
                <h3>Fitness Score</h3>
                <div class="animated-number">🎯 {fitness_score}</div>
                <p>Overall Rating</p>
            </div>
        """, unsafe_allow_html=True)

    # Interactive Visualizations
    st.markdown("### 📈 Workout Analytics")
    viz_cols = st.columns(2)
    
    with viz_cols[0]:
        # Calories burned trend
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=fitness_data['Session_Duration (hours)'],
            y=fitness_data['Calories_Burned'],
            mode='markers',
            name='Other Users',
            marker=dict(color='lightgray', size=8)
        ))
        fig.add_trace(go.Scatter(
            x=[duration],
            y=[predicted_calories],
            mode='markers',
            name='You',
            marker=dict(color='red', size=15, symbol='star')
        ))
        fig.update_layout(
            title='Your Performance vs Others',
            xaxis_title='Workout Duration (hours)',
            yaxis_title='Calories Burned',
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)

    with viz_cols[1]:
        # Heart rate zones analysis
        hr_zones = {
            'Peak': max_bpm,
            'Cardio': avg_bpm,
            'Fat Burn': (avg_bpm + resting_bpm) / 2,
            'Rest': resting_bpm
        }
        fig = go.Figure(go.Bar(
            x=list(hr_zones.values()),
            y=list(hr_zones.keys()),
            orientation='h',
            marker_color=['#ff4b4b', '#ff9f40', '#4CAF50', '#2196F3']
        ))
        fig.update_layout(
            title='Heart Rate Zones Analysis',
            xaxis_title='BPM',
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)

with tabs[1]:  # Detailed Analysis
    st.markdown("### 🔍 Detailed Fitness Analysis")
    
    # Initialize workout history if not exists
    if 'workout_history' not in st.session_state:
        st.session_state.workout_history = []

    # Add current workout to history
    current_workout = {
        'date': datetime.now(),
        'calories': predicted_calories,
        'duration': duration,
        'type': workout_type
    }

    # Only append if it's a new workout (you might want to add your own logic here)
    st.session_state.workout_history.append(current_workout)

    # Create DataFrame from history
    history_df = pd.DataFrame(st.session_state.workout_history)

    # Progress charts
    if not history_df.empty:
        fig = px.line(history_df, x='date', y='calories',
                     title='Calories Burned Over Time')
        st.plotly_chart(fig, use_container_width=True)

    # Workout type distribution
    if not history_df.empty:
        type_counts = history_df['type'].value_counts().reset_index()
        type_counts.columns = ['Workout_Type', 'Count']  # Rename columns
        
        fig = px.pie(
            type_counts, 
            values='Count',
            names='Workout_Type',
            title='Workout Type Distribution'
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No workout history available yet.")

with tabs[2]:  # Progress Tracking
    st.markdown("### 📊 Progress Dashboard")
    
    # Goal Setting
    if 'fitness_goals' not in st.session_state:
        st.session_state.fitness_goals = {
            'weight': weight,
            'calories': 500,
            'workouts_per_week': 3
        }
    
    # Goal progress visualization
    goals_col1, goals_col2 = st.columns(2)
    
    with goals_col1:
        st.markdown("#### Weight Goal Progress")
        weight_progress = (st.session_state.fitness_goals['weight'] - weight) / \
                         st.session_state.fitness_goals['weight'] * 100
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=abs(weight_progress),
            title={'text': "Progress to Goal"},
            gauge={'axis': {'range': [None, 100]}}
        ))
        st.plotly_chart(fig, use_container_width=True)

    with goals_col2:
        st.markdown("#### Workout Frequency Progress")
        frequency_progress = frequency / st.session_state.fitness_goals['workouts_per_week'] * 100
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=frequency_progress,
            title={'text': "Weekly Target"},
            gauge={'axis': {'range': [None, 100]}}
        ))
        st.plotly_chart(fig, use_container_width=True)

with tabs[3]:  # Recommendations
    st.markdown("### 💡 Personalized Recommendations")
    
    # Generate personalized recommendations
    recommendations = []
    
    # BMI-based recommendations
    if bmi < 18.5:
        recommendations.append({
            'category': 'Weight Management',
            'recommendation': 'Focus on strength training and increasing caloric intake',
            'priority': 'High'
        })
    elif bmi > 25:
        recommendations.append({
            'category': 'Weight Management',
            'recommendation': 'Incorporate more cardio and maintain caloric deficit',
            'priority': 'High'
        })
    
    # Heart rate-based recommendations
    if resting_bpm > 80:
        recommendations.append({
            'category': 'Cardiovascular Health',
            'recommendation': 'Increase low-intensity cardio sessions',
            'priority': 'Medium'
        })
    
    # Workout frequency recommendations
    if frequency < 3:
        recommendations.append({
            'category': 'Training Schedule',
            'recommendation': 'Gradually increase workout frequency to 3-4 times per week',
            'priority': 'Medium'
        })
    
    # Display recommendations
    for rec in recommendations:
        st.markdown(f"""
            <div class="insight-box">
                <h4>{rec['category']} ({rec['priority']} Priority)</h4>
                <p>{rec['recommendation']}</p>
            </div>
        """, unsafe_allow_html=True)

with tabs[4]:  # Health Insights
    st.markdown("### 🏥 Health Insights")
    
    # Calculate health metrics
    max_hr = 220 - age
    target_hr_zone = f"{int(max_hr * 0.64)} - {int(max_hr * 0.76)}"
    
    # Display health insights
    health_cols = st.columns(3)
    
    with health_cols[0]:
        st.markdown("""
            <div class="metric-container">
                <h3>Target Heart Rate Zone</h3>
                <p>For optimal fat burning:</p>
                <div class="animated-number">❤️ {}</div>
            </div>
        """.format(target_hr_zone), unsafe_allow_html=True)
    
    with health_cols[1]:
        daily_calories = predicted_calories * frequency / 7
        st.markdown("""
            <div class="metric-container">
                <h3>Daily Calorie Burn</h3>
                <p>From workouts:</p>
                <div class="animated-number">🔥 {:.0f}</div>
            </div>
        """.format(daily_calories), unsafe_allow_html=True)
    
    with health_cols[2]:
        weekly_active_minutes = duration * 60 * frequency
        st.markdown("""
            <div class="metric-container">
                <h3>Weekly Active Minutes</h3>
                <p>Total workout time:</p>
                <div class="animated-number">⏱️ {:.0f}</div>
            </div>
        """.format(weekly_active_minutes), unsafe_allow_html=True)

# Footer with additional information
st.markdown("""
    <div style='text-align: center; margin-top: 30px; padding: 20px; background: rgba(255,255,255,0.1);'>
        <p>Track your fitness journey and stay motivated! 💪</p>
        <p style='font-size: 0.8em;'>Data is updated in real-time based on your inputs</p>
    </div>
""", unsafe_allow_html=True)

# Initialize database and import data if needed
if 'db_initialized' not in st.session_state:
    if create_table():
        import_csv_to_db()
        st.session_state.db_initialized = True

# Load data from database
fitness_data = load_data()

# Check if data was loaded successfully
if fitness_data is None:
    st.error("Failed to load data from database")
    st.stop()

# Clean and preprocess the data
def clean_fitness_data(df):
    # Remove any rows with missing values in critical columns
    critical_columns = ['Weight (kg)', 'Height (m)', 'Resting_BPM', 'Calories_Burned']
    df = df.dropna(subset=critical_columns)
    
    # Replace any remaining NaN values with column means
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
    
    return df

# Clean the data
fitness_data = clean_fitness_data(fitness_data)

# Verify data is clean
if fitness_data.empty:
    st.error("No valid data available after cleaning")
    st.stop()

# Data Preprocessing
fitness_data = fitness_data.dropna()

# Define features and target variables
features = ['Age', 'Gender', 'Weight (kg)', 'Height (m)', 'Max_BPM', 'Avg_BPM', 
           'Resting_BPM', 'Session_Duration (hours)', 'Workout_Frequency (days/week)']
X = fitness_data[features]
y_calories = fitness_data['Calories_Burned']

# Create the pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['Age', 'Weight (kg)', 'Height (m)', 'Max_BPM', 'Avg_BPM', 
                                 'Resting_BPM', 'Session_Duration (hours)', 'Workout_Frequency (days/week)']),
        ('cat', OneHotEncoder(), ['Gender'])
    ])

calories_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train the model
calories_pipeline.fit(X, y_calories)

# Function to create styled visualizations
def create_styled_plot(data, x, y, title, color_palette=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if color_palette is None:
        color_palette = ['#4d9fff', '#ff4b4b']
    
    # Create scatter plot
    sns.scatterplot(data=data, x=x, y=y, hue='Gender', 
                    palette=color_palette, alpha=0.6, ax=ax)
    
    # Add trend line
    sns.regplot(data=data, x=x, y=y, scatter=False, 
                color='#333333', line_kws={'alpha': 0.5}, ax=ax)
    
    # Styling
    ax.set_title(title, fontsize=14, pad=20)
    ax.set_xlabel(x, fontsize=12)
    ax.set_ylabel(y, fontsize=12)
    
    # Grid styling
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Set background color
    ax.set_facecolor('#f8f9fa')
    fig.patch.set_facecolor('#f8f9fa')
    
    plt.tight_layout()
    return fig

# Function to create distribution plots
def create_distribution_plot(data, column, title):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.histplot(data=data, x=column, hue='Gender', 
                multiple="stack", palette=['#4d9fff', '#ff4b4b'], ax=ax)
    
    # Styling
    ax.set_title(title, fontsize=14, pad=20)
    ax.set_facecolor('#f8f9fa')
    fig.patch.set_facecolor('#f8f9fa')
    
    # Grid styling
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    return fig

# Streamlit UI components
st.title('Advanced Fitness Analytics Dashboard')

# Sidebar for user input
st.sidebar.header('Enter Your Metrics')

# User input fields
age = st.sidebar.number_input('Age', 18, 80, 30, key='sidebar_age')
gender = st.sidebar.selectbox('Gender', ['Male', 'Female'], key='sidebar_gender')
weight = st.sidebar.number_input('Weight (kg)', 40.0, 150.0, 70.0, key='sidebar_weight')
height = st.sidebar.number_input('Height (m)', 1.4, 2.2, 1.7, key='sidebar_height')
max_bpm = st.sidebar.number_input('Max Heart Rate (BPM)', 120, 200, 180, key='sidebar_max_bpm')
avg_bpm = st.sidebar.number_input('Average Heart Rate (BPM)', 60, 180, 140, key='sidebar_avg_bpm')
resting_bpm = st.sidebar.number_input('Resting Heart Rate (BPM)', 40, 100, 60, key='sidebar_resting_bpm')
duration = st.sidebar.number_input('Session Duration (hours)', 0.25, 4.0, 1.0, key='sidebar_duration')
frequency = st.sidebar.number_input('Workout Frequency (days/week)', 1, 7, 3, key='sidebar_frequency')

# Create input DataFrame
input_df = pd.DataFrame({
    'Age': [age],
    'Gender': [gender],
    'Weight (kg)': [weight],
    'Height (m)': [height],
    'Max_BPM': [max_bpm],
    'Avg_BPM': [avg_bpm],
    'Resting_BPM': [resting_bpm],
    'Session_Duration (hours)': [duration],
    'Workout_Frequency (days/week)': [frequency]
})

# Generate predictions
predicted_calories = calories_pipeline.predict(input_df)[0]

# Display predictions
st.header('Predictions and Analysis')
st.metric("Predicted Calories Burned", f"{predicted_calories:.0f} kcal")

# Function to calculate percentile and get comparative analysis
def get_metric_analysis(value, column, data):
    percentile = stats.percentileofscore(data[column], value)
    
    if column == 'Age':
        category = 'younger' if percentile > 50 else 'older'
    elif column in ['Weight (kg)', 'Height (m)', 'BMI']:
        category = 'higher' if percentile > 50 else 'lower'
    else:
        category = 'better' if percentile > 50 else 'lower'
        
    return percentile, category

# Create analysis blocks
st.header('Your Fitness Analysis 📊')

# Create three columns for metrics
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
        <div style='background-color: #f0f8ff; padding: 20px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
            <h3 style='color: #2c3e50;'>Age Analysis 👤</h3>
        """, unsafe_allow_html=True)
    
    age_percentile, age_category = get_metric_analysis(age, 'Age', fitness_data)
    avg_age = fitness_data['Age'].mean()
    
    st.markdown(f"""
        <div style='font-size: 1.1em;'>
            <p>Your age: <strong>{age} years</strong></p>
            <p>You are {age_category} than {age_percentile:.1f}% of users</p>
            <p>Average user age: {avg_age:.1f} years</p>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("""
        <div style='background-color: #f0fff0; padding: 20px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
            <h3 style='color: #2c3e50;'>Heart Rate Metrics ❤️</h3>
        """, unsafe_allow_html=True)
    
    max_bpm_percentile, max_bpm_category = get_metric_analysis(max_bpm, 'Max_BPM', fitness_data)
    avg_max_bpm = fitness_data['Max_BPM'].mean()
    
    st.markdown(f"""
        <div style='font-size: 1.1em;'>
            <p>Your max BPM: <strong>{max_bpm}</strong></p>
            <p>You are {max_bpm_category} than {max_bpm_percentile:.1f}% of users</p>
            <p>Average max BPM: {avg_max_bpm:.1f}</p>
            <p>Resting BPM: <strong>{resting_bpm}</strong></p>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col3:
    st.markdown("""
        <div style='background-color: #fff0f0; padding: 20px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
            <h3 style='color: #2c3e50;'>Body Metrics 💪</h3>
        """, unsafe_allow_html=True)
    
    weight_percentile, weight_category = get_metric_analysis(weight, 'Weight (kg)', fitness_data)
    bmi = weight / (height ** 2)
    
    # BMI categories
    bmi_category = (
        'Underweight' if bmi < 18.5 else
        'Normal weight' if bmi < 24.9 else
        'Overweight' if bmi < 29.9 else
        'Obese'
    )
    
    st.markdown(f"""
        <div style='font-size: 1.1em;'>
            <p>Your weight: <strong>{weight:.1f} kg</strong></p>
            <p>You are {weight_category} than {weight_percentile:.1f}% of users</p>
            <p>BMI: <strong>{bmi:.1f}</strong> ({bmi_category})</p>
            <p>Height: <strong>{height:.2f} m</strong></p>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Workout Analysis Section
st.header('Workout Analysis 🏋️‍♂️')
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
        <div style='background-color: #fff5e6; padding: 20px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
            <h3 style='color: #2c3e50;'>Session Analysis ⏱️</h3>
        """, unsafe_allow_html=True)
    
    duration_percentile, duration_category = get_metric_analysis(duration, 'Session_Duration (hours)', fitness_data)
    avg_duration = fitness_data['Session_Duration (hours)'].mean()
    
    st.markdown(f"""
        <div style='font-size: 1.1em;'>
            <p>Your session duration: <strong>{duration:.2f} hours</strong></p>
            <p>You train {duration_category} than {duration_percentile:.1f}% of users</p>
            <p>Average session duration: {avg_duration:.2f} hours</p>
            <p>Weekly frequency: <strong>{frequency} days</strong></p>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("""
        <div style='background-color: #e6f3ff; padding: 20px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
            <h3 style='color: #2c3e50;'>Calories Analysis 🔥</h3>
        """, unsafe_allow_html=True)
    
    avg_calories = fitness_data['Calories_Burned'].mean()
    calories_percentile = stats.percentileofscore(fitness_data['Calories_Burned'], predicted_calories)
    
    st.markdown(f"""
        <div style='font-size: 1.1em;'>
            <p>Predicted calories: <strong>{predicted_calories:.0f} kcal</strong></p>
            <p>This is higher than {calories_percentile:.1f}% of workouts</p>
            <p>Average calories burned: {avg_calories:.0f} kcal</p>
            <p>Calories/hour: <strong>{(predicted_calories/duration):.0f} kcal/hr</strong></p>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Add recommendations based on analysis
st.header('Personalized Recommendations 📋')

recommendations = []
if bmi > 25:
    recommendations.append("Consider focusing on cardio exercises to help manage weight")
if max_bpm > 180:
    recommendations.append("Your maximum heart rate is high. Consider monitoring intensity")
if duration < 1.0:
    recommendations.append("Try increasing your workout duration gradually")
if frequency < 3:
    recommendations.append("Consider increasing workout frequency to 3-4 times per week")
if resting_bpm > 80:
    recommendations.append("Your resting heart rate is elevated. Consider more cardiovascular training")

if recommendations:
    for i, rec in enumerate(recommendations, 1):
        st.markdown(f"""
            <div style='background-color: #f8f9fa; padding: 10px; border-left: 4px solid #007bff; margin-bottom: 10px;'>
                {i}. {rec}
            </div>
        """, unsafe_allow_html=True)
else:
    st.markdown("""
        <div style='background-color: #f8f9fa; padding: 10px; border-left: 4px solid #28a745;'>
            Great job! Your fitness metrics are within healthy ranges. Keep up the good work!
        </div>
    """, unsafe_allow_html=True)

# Create visualization section
st.header('Fitness Analytics Visualizations')

col1, col2 = st.columns(2)

with col1:
    calories_weight_plot = create_styled_plot(
        fitness_data, 
        'Weight (kg)', 
        'Calories_Burned',
        'Weight vs Calories Burned'
    )
    st.pyplot(calories_weight_plot)

with col2:
    calories_duration_plot = create_styled_plot(
        fitness_data,
        'Session_Duration (hours)',
        'Calories_Burned',
        'Duration vs Calories Burned',
        ['#3ed160', '#ff4b4b']
    )
    st.pyplot(calories_duration_plot)

# Distribution Analysis
st.header('Distribution Analysis')

col1, col2 = st.columns(2)

with col1:
    bpm_dist_plot = create_distribution_plot(
        fitness_data,
        'Avg_BPM',
        'Heart Rate Distribution'
    )
    st.pyplot(bpm_dist_plot)

with col2:
    duration_dist_plot = create_distribution_plot(
        fitness_data,
        'Session_Duration (hours)',
        'Workout Duration Distribution'
    )
    st.pyplot(duration_dist_plot)

# Clear matplotlib figures to free memory
plt.close('all')
