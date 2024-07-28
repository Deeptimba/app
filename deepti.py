import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Title of the app
st.title("Data Analysis and Visualization App")

# Developer name
st.markdown("**Developed by Deepti Alive**")

# Upload CSV file
st.sidebar.header("Upload your CSV file")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Read the uploaded CSV file
    data = pd.read_csv(uploaded_file)
    
    # Display the first few rows of the dataframe
    st.header("Data Preview")
    st.write(data.head())
    
    # Show summary statistics
    st.header("Summary Statistics")
    st.write(data.describe())
    
    # Select columns for visualization
    st.sidebar.header("Select Columns for Visualization")
    columns = data.columns.tolist()
    x_axis = st.sidebar.selectbox("Select X-axis", columns)
    y_axis = st.sidebar.selectbox("Select Y-axis", columns)
    
    # Plot the selected columns
    st.header("Scatter Plot")
    fig, ax = plt.subplots()
    ax = sns.scatterplot(x=data[x_axis], y=data[y_axis])
    st.pyplot(fig)
    
    # Correlation heatmap
    st.header("Correlation Heatmap")
    fig, ax = plt.subplots()
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)
    
    # Pairplot for selected columns
    st.header("Pairplot")
    selected_columns = st.sidebar.multiselect("Select Columns for Pairplot", columns)
    if selected_columns:
        fig = sns.pairplot(data[selected_columns])
        st.pyplot(fig)
else:
    st.info("Please upload a CSV file to get started.")

# Add statistical models (Example: Linear Regression)
if uploaded_file is not None:
    st.sidebar.header("Statistical Models")
    model_type = st.sidebar.selectbox("Select Model", ["None", "Linear Regression"])

    if model_type == "Linear Regression":
        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_squared_error, r2_score

        # Select columns for the model
        feature_cols = st.sidebar.multiselect("Select Feature Columns", columns)
        target_col = st.sidebar.selectbox("Select Target Column", columns)

        if feature_cols and target_col:
            X = data[feature_cols]
            y = data[target_col]

            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            # Create and train the model
            model = LinearRegression()
            model.fit(X_train, y_train)

            # Make predictions
            y_pred = model.predict(X_test)

            # Display the results
            st.header("Linear Regression Results")
            st.write("Mean Squared Error:", mean_squared_error(y_test, y_pred))
            st.write("R^2 Score:", r2_score(y_test, y_pred))

            # Plot the results
            st.header("Actual vs Predicted")
            fig, ax = plt.subplots()
            ax.scatter(y_test, y_pred)
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
            ax.set_xlabel('Actual')
            ax.set_ylabel('Predicted')
            st.pyplot(fig)
