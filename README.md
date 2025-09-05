# AI-Powered Robo-Advisory Platform

This project is a sophisticated robo-advisory platform that provides dynamic, personalized, and SEBI-compliant investment recommendations. It leverages a machine learning model to generate portfolio advice based on user profiles and includes features for automatic rebalancing and regulatory compliance.

## Features

*   **AI-Powered Recommendations**: Utilizes a `RandomForestClassifier` to offer personalized portfolio recommendations based on user's age, income, financial goals, and risk appetite.
*   **Dynamic Portfolios**: Generates dynamic portfolio allocations for different risk profiles, such as "Aggressive Growth," "Balanced," and "Conservative."
*   **Automatic Rebalancing**: Simulates market fluctuations and includes a feature to check if the user's portfolio has deviated from its target allocation, recommending rebalancing when necessary.
*   **SEBI Compliance**:
    *   **Record-Keeping**: Logs all client interactions and investment recommendations to `client_interactions.log` for regulatory auditing, in line with SEBI's five-year data retention policy.
    *   **Disclosure**: Provides clear explanations of how the AI model works, its limitations, and the importance of consulting a registered financial advisor.

## Technologies Used

*   **Python**: The core programming language for the application.
*   **Streamlit**: For building the interactive web application.
*   **Scikit-learn**: For implementing the machine learning model.
*   **Pandas**: For data manipulation and creating the mock dataset.
*   **Numpy**: For numerical operations, especially in the rebalancing simulation.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: A `requirements.txt` file is not provided, but you can install the dependencies directly:*
    ```bash
    pip install streamlit scikit-learn pandas numpy
    ```

## Usage

To run the application, execute the following command in your terminal:

```bash
streamlit run project.py
```

This will start the Streamlit server and open the application in your web browser. You can then interact with the sidebar to input your profile details and receive an AI-powered investment recommendation.
