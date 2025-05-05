# ASER
Optimizing Volunteer Deployment to Improve Foundational Learning Outcomes in Rural India

# Optimizing Volunteer Deployment for Foundational Learning in Rural India

## Project Overview

This project develops and implements a data-driven approach to optimize the allocation of limited volunteer teaching hours across districts within various Indian states. The goal is to maximize the potential impact on foundational learning outcomes (reading and numeracy) by prioritizing districts with the greatest need, based on public data from the Annual Status of Education Report (ASER).

## Motivation

India faces significant challenges in ensuring foundational literacy and numeracy for all children. While volunteer programs offer valuable support, their resources are finite. This project addresses the need for a systematic, evidence-based method to deploy these resources effectively, moving beyond ad-hoc allocations to target interventions where they are most needed, considering the disparities between districts.

## Methodology

The core of this project is a **state-wise Linear Programming (LP) optimization model** implemented in Python (`district2.py`). The script performs the following key steps:

1.  **Data Aggregation & Cleaning:**
    *   Reads multiple state-specific ASER district data CSV files from a specified directory.
    *   Identifies states and attempts to filter out state-level summary rows.
    *   Converts proficiency data to numeric types.
    *   **Crucially, removes districts with missing or invalid (<=0) proficiency data** for reading (and math, if composite score is used) to ensure data integrity.
2.  **Need Score Calculation:**
    *   Calculates inverse proficiency scores (`g_read_inv`, `g_math_inv`) for valid districts, handling near-zero values safely.
    *   Computes a `need_score` based on configuration:
        *   **Composite Score (Default):** Uses a weighted average of reading and math inverse scores (e.g., 50% reading, 50% math).
        *   **Reading Only:** Uses only the reading inverse score.
3.  **State-wise Optimization:**
    *   Iterates through each unique state present in the cleaned data.
    *   For each state, solves an LP problem using `scipy.optimize.linprog`.
    *   **Objective:** Maximize the sum of (allocated hours * `need_score`) across districts within the state. This proxies maximizing expected learning impact.
    *   **Constraints:**
        *   Total available hours per state (calculated based on `AVG_HOURS_PER_DISTRICT` and number of districts in the state).
        *   Minimum hours per district (`MIN_HOURS_PER_DISTRICT`).
        *   Maximum hours per district (`MAX_HOURS_PER_DISTRICT`).
4.  **Output Generation:**
    *   Saves the final combined data, including the calculated need scores and optimized hours (`Optimized_Hours_Statewise`), to a CSV file.
    *   Generates several plots visualizing the data distribution and optimization results.

## Data

*   **Source:** District-level data from the Annual Status of Education Report (ASER) for **[Specify ASER Year, e.g., 2022]**. Data can be found at the [ASER Centre website](https://www.asercentre.org/).
*   **Input Format:** The script expects multiple `.csv` files, typically one per state, placed inside the directory specified by the `CSV_DIRECTORY` variable in the script.
*   **Required Columns:** Ensure your CSV files contain columns with the *exact* names specified by `DISTRICT_COLUMN`, `READING_PROFICIENCY_COLUMN`, and (if using composite score) `MATH_PROFICIENCY_COLUMN` in the script configuration section.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone [Your Repository URL]
    cd [Your Repository Name]
    ```
2.  **Python:** Ensure you have Python 3.x installed.
3.  **Install Dependencies:** Install the required Python libraries. It's recommended to use a virtual environment.
    ```bash
    pip install pandas numpy scipy matplotlib seaborn
    ```
    *(Optional but recommended: Create a `requirements.txt` file listing these dependencies and use `pip install -r requirements.txt`)*

## Usage

1.  **Place Data:** Put all your state-specific ASER CSV files into a single directory (e.g., create a folder named `data` inside the repository).
2.  **Configure Script (`district2.py`):**
    *   Set `CSV_DIRECTORY` to the path of the folder containing your data (e.g., `CSV_DIRECTORY = 'data'`).
    *   Verify `DISTRICT_COLUMN`, `READING_PROFICIENCY_COLUMN`, and `MATH_PROFICIENCY_COLUMN` match your CSV headers exactly.
    *   Choose your **Optimization Strategy**:
        *   Set `USE_COMPOSITE_SCORE` (`True` or `False`).
        *   If `True`, adjust `WEIGHT_READING` and `WEIGHT_MATH` if desired (ensure they sum to 1.0).
    *   Adjust **Resource Allocation Parameters** if needed:
        *   `AVG_HOURS_PER_DISTRICT` (influences state budgets).
        *   `MIN_HOURS_PER_DISTRICT` (baseline allocation).
        *   `MAX_HOURS_PER_DISTRICT` (capacity cap).
3.  **Run Script:**
    ```bash
    python district2.py
    ```
4.  **Check Outputs:** The script will print status messages and summaries to the console. Check the directory where you ran the script for the following output files:
    *   `districts_statewise_optimized_allocation_final.csv`: The main output CSV with results.
    *   `dist_reading_proficiency_histogram_cleaned.png`: Histogram of reading scores.
    *   `dist_lowest_reading_proficiency_barchart_cleaned.png`: Bar chart of lowest proficiency districts.
    *   `dist_optimized_hours_histogram_statewise.png`: Histogram of allocated volunteer hours.
    *   `dist_highest_allocated_hours_barchart_statewise.png`: Bar chart of districts receiving most hours.
    *   `dist_hours_vs_proficiency_scatter_statewise.png`: Scatter plot of hours vs. reading proficiency.

## Key Results Summary

The state-wise optimization model successfully generates allocation plans prioritizing districts with lower foundational learning levels (based on the chosen need score). Key observations include:

*   **Targeted Allocation:** Districts with higher calculated `need_score` consistently receive more volunteer hours, up to the maximum limit.
*   **"Bang-Bang" Distribution:** The linear programming approach often results in most districts receiving either the minimum (1.0) or maximum (5.0) allowed hours, reflecting efficient resource allocation towards the highest-need areas under the model's constraints.
*   **State Budget Impact:** Intermediate hour allocations occur primarily when the total calculated budget for a state is exhausted before all high-need districts reach the maximum cap.

Refer to the generated plots and the output CSV (`districts_statewise_optimized_allocation_final.csv`) for detailed district-level results.

## Limitations

*   **Data Granularity:** District-level averages mask intra-district variations.
*   **Model Assumptions:** Assumes linear impact, uses inverse proficiency as a proxy for expected gain, and results are sensitive to parameter choices (bounds, budget, weights).
*   **Excluded Factors:** Does not account for real-world constraints like travel feasibility, local implementation capacity, volunteer skills, or socio-economic factors.
*   **State-wise Scope:** Optimizes within states, not necessarily allocating based on absolute need across the entire country.

(Please refer to the full project report for a detailed discussion of limitations).

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## Author

*   [Your Name] - Indian Statistical Institute, Kolkata

## Acknowledgements

*   Data sourced from the Annual Status of Education Report (ASER), ASER Centre (\url{https://www.asercentre.org/}).
*   Developed using Python and libraries including Pandas, NumPy, SciPy, Matplotlib, and Seaborn.
